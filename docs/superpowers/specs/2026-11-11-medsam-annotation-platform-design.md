# 医学影像辅助标注平台设计

日期：2026-11-11  
状态：已于 2026-11-11 确认。后端到导出的实现计划见 `docs/superpowers/plans/2026-11-11-medsam-annotation-backend.md`。界面、公开数据评测和微调各自另写计划。

## 意图

做一个面向互联网大厂 AI 产品经理作品的 ToB 系统：医学影像辅助标注平台。

使用方是标注团队和算法团队，不是临床医生。标注员上传影像、画一个框，MedSAM 给出掩膜；人接受或改完后，这条结果才进入可训练数据。算法或产品侧看的是预标注够不够用：和人工结果差多少、花了多久、哪些图失败。后面在 AutoDL 上只用人工确认过的数据微调，并和微调前对比。

作品要证明的是这条业务链路，而不是“把 MedSAM 的 demo 跑通”。

## 规模

这是一个完整项目，按五个阶段做完：数据与两个服务、标注界面、导出与人工修正指标、公开数据评测、AutoDL 微调对比。时间按做扎实来排，不按演示稿压缩。第一阶段结束时，不打开网页也能用接口走完一次真实推理并落库。

## 目标

1. 业务服务和 MedSAM 推理服务分开。网页和接口都不把掩膜写死。
2. 一条任务里，模型预测和人工结果是两条记录。只有人工接受或修改后的结果可以导出用于微调。
3. 推理从第一天起就是任务：提交、查状态、取结果。请求线程不一直占着直到模型跑完。
4. 本地用 Docker Compose 跑 PostgreSQL 和 Redis，业务服务、推理服务和 worker 用 CPU 走通。AutoDL 上用同一份 Compose，推理和训练改用 GPU。
5. 元数据进 PostgreSQL。影像和掩膜进共享文件夹。
6. 标注界面能上传、画框、看掩膜、用笔刷改、接受。
7. 评测用公开、带金标准掩膜的二维影像，提示框由金标准外接矩形生成。报告 Dice 和耗时，并按数据集分开。
8. 微调只用导出的人工结果。报告微调前后的 Dice。
9. 配套一份产品说明：用户、流程、指标、建议的启用门槛。门槛是待验证的假设，不写成已经成立的临床结论。

## 非目标

- 不做多租户、SSO、细粒度权限、审计平台。
- 不复刻 CVAT 或 Label Studio 的全部能力：视频、共识标注、Webhook、对象存储、云盘同步都不做。
- 不做临床诊断，不接 PACS，不处理 DICOM 服务器。第一版只处理普通二维图像（PNG、JPEG）。
- 不做三维体数据，不接 MedSAM2。
- 不把官方训练脚本改写成另一套框架。导出先写成图像、掩膜和清单；微调阶段再适配官方训练输入。
- 不做移动端，不做对外网开放的部署。

## 明确不要做成的样子

- 只有前端、掩膜来自假数据。
- 简历上只能写“复现了 MedSAM”。
- 平台功能做了很多，评测和微调没有发生。
- 把 Dice 或速度写成可以给医生做诊断的效果。
- 使用任何非公开的病人影像。

## 约束

- 开发机是 Windows，没有 NVIDIA 显卡。本地推理已经用官方 demo 图和 CPU 跑通过，权重在 `work_dir/MedSAM/medsam_vit_b.pth`。
- PostgreSQL 和 Redis 用 Docker Compose 跑。开发机需要安装 Docker Desktop（含它要求的 WSL2）。工具还没装时，让用户安装，不改成免 Docker 的替代数据库。
- 现有仓库是 MedSAM 官方代码。新产品放在 `platform/`，不改官方模型结构文件。推理服务调用现有的 `segment_anything`。
- 本机 Python 是 3.14。已有 `.venv` 里是 CPU 版 PyTorch，推理服务继续用它。业务服务、推理服务和 worker 跑在宿主机上，以便直接用这个环境和权重。
- GPU 训练和更快的推理放在 AutoDL。两边通过环境变量切换设备，不维护两份业务代码。
- 业务服务只认 `DATABASE_URL`。数据库和缓存由 Compose 提供，只把端口绑在 `127.0.0.1`。
- 服务只监听本机。

## 方案

考虑过三种做法。

1. **自己做一套小系统，结构对齐成熟标注平台。** 业务 API 与模型服务分离；预测和人工标注分表；耗时推理走任务状态。这是选定方案。原因：作品的价值在自己把“预标注、人工确认、质量门槛、再用确认数据微调”做出来，同时避免从零发明一套和行业习惯相反的结构。
2. **基于 MONAI Label。** 医学影像接口现成，但任务、质量统计和产品流程要迁就它，多人保存也缺少版本。没有选。
3. **在 CVAT 或 Label Studio 上只接 MedSAM。** 最接近公司现成采购，但作品会变成集成。没有选。

选定方案吸收的具体习惯：

- Label Studio：项目、任务、模型预测、人工标注分开。
- MONAI Label：模型输出和最终标注分开保存；只有最终标注进入训练。
- CVAT：自动标注是独立任务，客户端拿任务号轮询。
- Label Studio 的任务接口：提交后立刻返回，客户端轮询状态。本项目从第一天起用 Redis worker 执行推理，不把“先在请求进程里跑”当成正式路径。测试里可以用假的推理客户端，不连接 Redis。

不照搬的部分：登录中心、策略引擎、分析型数据库、按职能拆开的多种 worker、对象存储。

## 已经定下来的产品决定

- 场景是通用二维医学影像，不限定病理。
- 作品面向互联网大厂 AI 产品经理岗位。
- 使用方是 ToB 标注和算法团队。
- 必须有真实后端。
- 采用上面的第 1 种方案。
- 时间和算力可以投入，优先含金量和业务贴近度。
- 代码放在当前 MedSAM 项目目录的 `platform/` 下。

## 系统边界

```
浏览器
  → 业务服务 platform/app（项目、任务、标注、导出、任务状态）
      → PostgreSQL（元数据）
      → DATA_ROOT（原图、预测掩膜、人工掩膜、导出包）
      → Redis 队列 + platform/worker
          → 推理服务 platform/infer（只做 MedSAM 推理，稍后加训练入口）
```

两个服务必须能同时看到 `DATA_ROOT`。业务服务传给推理服务的是相对路径，不传本机绝对路径，这样换机器时只改根目录。

推理服务启动时加载一次权重。它不理解项目和用户。

## 数据模型

标识都用 UUID。时间用带时区的时间戳。

**projects**

- id，name，description
- label_name：这一项目在标什么，例如“病灶”。第一版掩膜仍是二值的，类别名只用于展示和导出
- created_at

**tasks**

- id，project_id
- filename，relative_path，width，height
- status：`pending`，`predicting`，`predicted`，`accepted`
- created_at

一张图一条任务。第一版一张图一次只提交一个框。表结构允许一个任务下有多条预测，后面可以加第二个框，而不必改预测表。

**jobs**

- id，type：`infer`，`export`，`train`
- status：`queued`，`running`，`succeeded`，`failed`
- task_id：导出和训练任务可为空
- project_id
- payload，result，error
- created_at，started_at，finished_at

**predictions**

- id，task_id，job_id
- box：左、上、右、下，原图像素坐标
- mask_relative_path
- model_version：权重文件名加一个固定版本字符串
- latency_ms
- created_at

**annotations**

- id，task_id，prediction_id（可空）
- mask_relative_path
- action：`accept` 或 `edit`
- dice_vs_prediction：有对应预测时计算
- annotator_name：默认 `annotator`
- created_at

同一任务可以有多条标注记录，最新的一条是当前结果。导出和指标都读最新一条。不覆盖旧记录。

**exports**

- id，project_id，job_id
- relative_path
- task_count
- created_at

删除项目时，数据库记录和 `DATA_ROOT` 下该项目的文件一起删。第一版可以先不做删除接口。

## 接口

业务服务提供 JSON API，浏览器使用同一来源上的页面。

- `POST /api/projects`，`GET /api/projects`，`GET /api/projects/{id}`
- `POST /api/projects/{id}/images`：multipart，可多张。每张建立一个 `pending` 任务
- `GET /api/projects/{id}/tasks`
- `GET /api/tasks/{id}`：含原图地址、最新预测、最新标注
- `POST /api/tasks/{id}/infer`：body 为 `{ "box": [x0, y0, x1, y1] }`。校验框在图像内且宽高为正。创建 `queued` 任务，立刻返回 `{ "job_id" }`。任务状态改为 `predicting`
- `GET /api/jobs/{id}`：返回状态、错误、成功后的预测或导出信息
- `POST /api/tasks/{id}/annotations`：`{ "action": "accept" }` 采用最新预测；或 `{ "action": "edit", "mask_png_base64": "..." }` 保存笔刷结果。成功后任务变为 `accepted`
- `POST /api/projects/{id}/exports`：只打包已接受任务，返回 job id
- `GET /api/files/{relative_path}`：只允许读取 `DATA_ROOT` 之内的文件

推理服务：

- `GET /health`
- `POST /v1/infer`：`{ "image_relative_path", "box", "output_relative_path" }`。返回宽、高和耗时。掩膜写成与原图同尺寸的单通道 PNG，前景为 255

失败时业务任务写成 `failed`，任务状态退回 `pending`；如果该任务已经有预测，退回 `predicted`。错误信息进 job.error，不把堆栈直接给页面。

## 界面

业务服务托管静态页面，第一版三屏。

1. 项目列表：新建项目，进入项目。
2. 项目页：上传图片，看任务状态，打开一张图。
3. 标注页：显示图像；拖拽画框；提交推理；轮询时显示进行中；叠加最新掩膜；笔刷可添加或擦除；接受预测，或保存修改。显示这一次的耗时，以及修改后和预测之间的 Dice。

笔刷只改掩膜，不改原图。不做多边形、关键点和分类面板。

## 导出与微调数据

导出目录：

- `images/`：原图
- `masks/`：最新人工掩膜
- `predictions/`：对应的模型掩膜，便于对比
- `manifest.jsonl`：task id、文件名、框、动作、dice、模型版本

没有人工标注的任务不进入导出。微调阶段读取这份导出，不直接读预测表。

## 评测

单独的评测脚本，不走标注界面。

对每张有金标准的图，用金标准掩膜的外接矩形作为提示框，调用已经落成的推理服务或同一套推理函数。记录 Dice 和耗时。

第一批数据用两个小的公开二维分割集，模态要不同，并且带研究可用的许可：

- Kvasir-SEG：内镜息肉
- BUSI：乳腺超声

数据放在 `DATA_ROOT/benchmarks/`，不提交进 Git。报告写明下载日期、样本数、是否用了全集、失败样本的例子。框来自金标准，所以这个 Dice 表示“人已经大致框出目标时，模型勾勒得怎么样”，不能写成无提示的全自动检出。

启用门槛先写成假设：人工修正相对预测的中位 Dice 达到 0.80，并且标注员愿意接受而不是重画，才考虑把该模型用作默认预标注。评测之后用真实数字接受或修改这个假设。

## 微调

在 AutoDL 上做，不在这台无显卡的电脑上训练。

用某一项目导出的人工掩膜，留出一部分任务不参与训练。比较微调前后在留出集上的 Dice。训练脚本放在 `platform/train/`，输入是导出目录。官方 `train_one_gpu.py` 只作为数据格式和超参的参照，不把整个官方多卡流程搬进产品。

模型版本写进预测记录。微调后的权重要换一个版本字符串，旧预测保留。

## 本地与 AutoDL

环境变量：

- `DATABASE_URL`
- `DATA_ROOT`
- `MEDSAM_CHECKPOINT`
- `INFER_URL`
- `DEVICE`：`cpu` 或 `cuda`
- `REDIS_URL`：worker 用它消费 `infer` 和 `train` 任务

`platform/docker-compose.yml` 提供 PostgreSQL 16 和 Redis 7，数据卷持久化，端口只绑定本机。本地和 AutoDL 用同一份 Compose。

本地启动：Compose，然后启动推理服务和业务服务。`REDIS_URL` 有值时，由 `platform/worker` 执行 `infer` 任务。这样本机和云上的任务路径一致。

AutoDL 上同样启动 Compose、推理服务、业务服务和 worker。worker 处理 `infer` 和 `train`。导出仍可在业务进程内做。`DEVICE=cuda`。

## 测试

- 业务接口测试使用假的推理客户端，不加载权重。覆盖：建项目、上传、非法框、任务从 queued 到 succeeded、接受、修改、只导出已接受任务、文件路径不能逃出 `DATA_ROOT`。
- 一个可选的真实推理测试：权重存在时，用 `assets/img_demo.png` 和官方示例框跑通。默认测试命令跳过它。

## 目录

```
platform/
  app/          业务服务和页面
  infer/        MedSAM 推理服务
  worker/       Redis worker
  train/        微调入口
  eval/         公开数据评测
  tests/
docs/superpowers/specs/   本设计
```

产品说明和评测报告在对应阶段再写，不在搭建第一阶段提前写满。

## 实现顺序

1. Compose 里的 PostgreSQL 和 Redis、表、两个服务、worker、demo 图经 API 出掩膜并落库。
2. 三屏界面：上传、画框、笔刷、接受。
3. 导出，以及预测和人工结果的 Dice。
4. Kvasir-SEG 与 BUSI 评测，写成报告。
5. AutoDL 微调、留出集对比、模型版本。

前一阶段没有真实跑通，不开始把下一阶段写进简历材料。

## 留给实现时决定的细节

- FastAPI 作为两个服务的框架。对照产品用 Django，是因为它们有大量内置应用；这里只有一个业务域，FastAPI 足够，而且推理服务保持轻量。
- 用 SQLAlchemy 和迁移工具管理表结构。
- PostgreSQL 16 和 Redis 7 的镜像、库名、用户名写在 Compose 里。开发库密码只用在本机 Compose，不写成生产密钥。
- 第一版没有登录。标注记录上的 `annotator_name` 来自请求里的可选字段，缺省为 `annotator`。
- 页面用普通 HTML、CSS 和少量 JavaScript，由业务服务托管。不引入前端构建工具。
- 笔刷在浏览器里改一张与原图同尺寸的掩膜，保存为 PNG。
- 推理服务一次只处理一个请求。本地 CPU 上官方 demo 大约需要几十秒，队列把这个等待留在任务状态里。
- Git 不提交权重、数据库文件、`DATA_ROOT` 和评测原图。

## 简历里可以声称的范围

可以写已经实现并跑出数字的流程、指标和取舍。不把未跑的微调写成完成。不把金标准外接框提示下的 Dice 写成无提示检测效果。不把本机 demo 写成临床可用。
