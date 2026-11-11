# MedSAM 辅助标注后端 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供业务 API、MedSAM 推理服务和 Redis worker，使一张示例图可以画框、异步出掩膜、人工接受或修改，并只导出已接受的任务。

**Architecture:** 业务服务负责项目、任务、预测、人工标注、导出和任务状态。推理服务只根据相对路径和框写掩膜。worker 从 Redis 队列取 `infer` 任务，调用推理服务，再把预测写入 PostgreSQL。浏览器界面、公开数据评测和 AutoDL 微调不在本计划内。

**Tech Stack:** FastAPI、SQLAlchemy、Alembic、PostgreSQL 16、Redis 7、RQ、httpx、Pillow、PyTorch（仅推理服务）。

**Spec:** `docs/superpowers/specs/2026-11-11-medsam-annotation-platform-design.md`

## Global Constraints

- 标识使用 UUID，时间使用带时区的时间戳。
- 任务状态只使用 `pending`、`predicting`、`predicted`、`accepted`。
- 任务类型只使用 `infer`、`export`、`train`。任务状态只使用 `queued`、`running`、`succeeded`、`failed`。
- 标注动作只使用 `accept` 和 `edit`。缺省标注人是 `annotator`。
- 推理请求体是 `{ "box": [x0, y0, x1, y1] }`，坐标是原图像素，且必须是整数。
- 推理服务 `POST /v1/infer` 的请求体是 `{ "image_relative_path", "box", "output_relative_path" }`，返回宽、高和耗时。掩膜是与原图同尺寸的单通道 PNG，前景为 255。
- 模型版本字符串是 `<权重文件名>:medsam-vit-b`。
- 失败时任务写成 `failed`。若该图还没有预测，任务退回 `pending`；若已有预测，退回 `predicted`。`job.error` 不包含堆栈。
- 文件接口只读取 `DATA_ROOT` 之内的文件。
- 没有人工标注的任务不进入导出。旧预测和旧标注不覆盖。
- 业务测试使用假的推理客户端，不加载权重。真实权重测试默认跳过。
- 权重、`.env`、`platform/data/` 不提交。官方已跟踪的 `assets/img_demo.png` 保持不动。
- 服务和 Compose 端口只绑定 `127.0.0.1`。
- 不实现登录、删除项目、页面、评测脚本和训练。

## Review Focus

- 已有预测之后再次推理失败，任务应回到 `predicted`，且错误文本不含 `Traceback`。
- 贴着图像边界的框有效；宽度或高度为零、或越界一个像素的框无效，且不创建任务。
- `..`、绝对路径和反斜杠都不能通过文件接口读出 `DATA_ROOT` 之外的内容。
- 没有预测时不能接受；修改要保留历史记录；没有预测时的修改，Dice 为空。
- 导出只包含 `accepted` 任务；同一张图的第二次预测新增一行，不替换第一次。

## File Structure

- `platform/docker-compose.yml`：PostgreSQL 16 与 Redis 7。
- `platform/requirements.txt`：后端依赖。推理依赖仍用仓库根目录已有的 `.venv`。
- `platform/.env.example`：环境变量样例。
- `platform/app/settings.py`：读取环境变量。
- `platform/app/db.py`：引擎、会话、`Base`。
- `platform/app/models.py`：六张表。
- `platform/app/schemas.py`：请求和响应模型。
- `platform/app/paths.py`：相对路径约束。
- `platform/app/dice.py`：Dice。
- `platform/app/main.py`：业务应用，端口 8000。
- `platform/app/routers/projects.py`：项目、上传、导出。
- `platform/app/routers/tasks.py`：任务、推理提交、标注。
- `platform/app/routers/jobs.py`：任务状态。
- `platform/app/routers/files.py`：文件读取。
- `platform/infer/segment.py`：MedSAM 调用。
- `platform/infer/main.py`：推理应用，端口 8001。
- `platform/worker/jobs.py`：消费一条任务并落库。
- `platform/worker/main.py`：RQ worker 入口。
- `platform/alembic/`：迁移。
- `platform/tests/`：接口测试。

后续计划才创建 `platform/app/static/`、`platform/eval/`、`platform/train/`。

---

### Task 1: Compose 与应用配置

**Files:**
- Create: `platform/docker-compose.yml`
- Create: `platform/requirements.txt`
- Create: `platform/.env.example`
- Create: `platform/app/settings.py`
- Create: `platform/app/main.py`
- Create: `platform/tests/test_health.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 无。
- Produces: `Settings` 字段 `database_url`、`data_root`、`medsam_checkpoint`、`infer_url`、`device`、`redis_url`。`create_app(settings: Settings) -> FastAPI`。默认值：`postgresql+psycopg://medsam:medsam@127.0.0.1:5432/medsam`，`platform/data`，`work_dir/MedSAM/medsam_vit_b.pth`，`http://127.0.0.1:8001`，`cpu`，`redis://127.0.0.1:6379/0`。

- [ ] **Step 1: Write the failing test**

```python
def test_health_and_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(data_root=tmp_path)
    assert settings.database_url.endswith("/medsam")
    assert settings.device == "cpu"
    response = TestClient(create_app(settings)).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_health.py -v`  
Expected: FAIL，`Settings` 或 `create_app` 不存在。

- [ ] **Step 3: Implement configuration and Compose**

`docker-compose.yml` 使用 `postgres:16` 和 `redis:7`。数据库名、用户名、密码都是 `medsam`。端口是 `127.0.0.1:5432:5432` 和 `127.0.0.1:6379:6379`。PostgreSQL 使用命名卷。`requirements.txt` 包含 fastapi、uvicorn、sqlalchemy、alembic、psycopg[binary]、pydantic-settings、httpx、rq、redis、fakeredis、pillow、numpy、pytest。`.gitignore` 增加 `platform/data/`。

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pip install -r platform/requirements.txt`  
然后重新运行 Step 2 的命令。Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/docker-compose.yml platform/requirements.txt platform/.env.example platform/app/settings.py platform/app/main.py platform/tests/test_health.py .gitignore
git commit -m "feat: add annotation platform service configuration"
```

### Task 2: 数据表

**Files:**
- Create: `platform/app/db.py`
- Create: `platform/app/models.py`
- Create: `platform/alembic.ini`
- Create: `platform/alembic/env.py`
- Create: `platform/alembic/versions/001_initial.py`
- Create: `platform/tests/test_models.py`

**Interfaces:**
- Consumes: `Settings.database_url`。
- Produces: `Project`、`Task`、`Job`、`Prediction`、`Annotation`、`Export`。字段名与设计文档的数据模型一致。`box` 用四个整数列 `x0,y0,x1,y1`。`payload`、`result` 用 JSON。`dice_vs_prediction` 可空。

- [ ] **Step 1: Write the failing test**

```python
def test_tables_match_spec(db):
    names = set(inspect(db.bind).get_table_names())
    assert names >= {"projects", "tasks", "jobs", "predictions", "annotations", "exports"}
    project = Project(name="肝脏", description="CT", label_name="病灶")
    db.add(project)
    db.flush()
    task = Task(project_id=project.id, filename="a.png", relative_path="p/a.png", width=4, height=4, status="pending")
    db.add(task)
    db.commit()
    assert task.id is not None
    assert task.created_at.tzinfo is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_models.py -v`  
Expected: FAIL，模型不存在。

- [ ] **Step 3: Implement the models and initial migration**

用 Alembic 生成初始迁移。测试夹具连接到 Compose 中的数据库；若 `127.0.0.1:5432` 连不上，测试跳过并提示先运行 `docker compose -f platform/docker-compose.yml up -d`。每个测试使用独立 schema 或事务回滚，不污染开发库。

- [ ] **Step 4: Run test to verify it passes**

先启动 Compose，再运行 Step 2 的命令。Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/db.py platform/app/models.py platform/alembic.ini platform/alembic platform/tests/test_models.py
git commit -m "feat: add annotation platform tables"
```

### Task 3: 项目接口

**Files:**
- Create: `platform/app/schemas.py`
- Create: `platform/app/routers/projects.py`
- Create: `platform/tests/test_projects.py`
- Modify: `platform/app/main.py`

**Interfaces:**
- Consumes: `Project`。
- Produces: `POST /api/projects` 接受 `name`、`description`、`label_name`，三者都是非空字符串。`GET /api/projects` 和 `GET /api/projects/{id}` 返回这些字段、`id` 和 `created_at`。未知项目返回 404。

- [ ] **Step 1: Write the failing test**

```python
def test_create_and_get_project(client):
    created = client.post("/api/projects", json={"name": "肝脏", "description": "CT", "label_name": "病灶"})
    assert created.status_code == 201
    body = created.json()
    assert body["label_name"] == "病灶"
    assert client.get(f"/api/projects/{body['id']}").json()["name"] == "肝脏"
    assert client.post("/api/projects", json={"name": "", "description": "CT", "label_name": "病灶"}).status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_projects.py -v`  
Expected: FAIL，路由不存在。

- [ ] **Step 3: Implement the project routes**

把路由挂到 `create_app`。列表按 `created_at` 升序。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/schemas.py platform/app/routers/projects.py platform/app/main.py platform/tests/test_projects.py
git commit -m "feat: add annotation project API"
```

### Task 4: 上传影像

**Files:**
- Create: `platform/tests/test_uploads.py`
- Modify: `platform/app/routers/projects.py`
- Modify: `platform/app/schemas.py`

**Interfaces:**
- Consumes: `Project`、`Task`、`Settings.data_root`。
- Produces: `POST /api/projects/{id}/images` 接收 multipart 字段 `files`，可多张。PNG 和 JPEG 各建一条 `pending` 任务。相对路径是 `projects/{project_id}/images/{task_id}{suffix}`。`GET /api/projects/{id}/tasks` 返回任务列表。`GET /api/tasks/{id}` 在没有结果时返回 `latest_prediction: null` 和 `latest_annotation: null`，`image_url` 为 `/api/files/` 加相对路径。

- [ ] **Step 1: Write the failing test**

```python
def test_upload_png_and_jpeg_and_reject_text(client, tmp_data):
    project_id = create_project(client)
    response = client.post(f"/api/projects/{project_id}/images", files=[
        ("files", ("gray.png", grayscale_png_bytes(3, 2), "image/png")),
        ("files", ("scan.jpg", jpeg_bytes(5, 4), "image/jpeg")),
    ])
    assert response.status_code == 201
    tasks = client.get(f"/api/projects/{project_id}/tasks").json()
    assert [task["status"] for task in tasks] == ["pending", "pending"]
    assert tasks[0]["width"] == 3 and tasks[0]["height"] == 2
    rejected = client.post(f"/api/projects/{project_id}/images", files=[("files", ("a.txt", b"hello", "text/plain"))])
    assert rejected.status_code == 415
    assert len(client.get(f"/api/projects/{project_id}/tasks").json()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_uploads.py -v`  
Expected: FAIL，上传路由不存在。

- [ ] **Step 3: Implement uploads**

只接受 `.png`、`.jpg`、`.jpeg`，并且文件能被 Pillow 打开。一张非法文件使整个请求失败，不留下新任务。灰度图保留原文件，宽高取图像尺寸。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/routers/projects.py platform/app/schemas.py platform/tests/test_uploads.py
git commit -m "feat: upload images into annotation tasks"
```

### Task 5: 文件读取约束

**Files:**
- Create: `platform/app/paths.py`
- Create: `platform/app/routers/files.py`
- Create: `platform/tests/test_files.py`
- Modify: `platform/app/main.py`

**Interfaces:**
- Consumes: `Settings.data_root`。
- Produces: `resolve_data_path(data_root: Path, relative_path: str) -> Path`。合法时返回 `data_root` 下的绝对路径；`..`、绝对路径、反斜杠和空路径抛出 `PathEscape`。`GET /api/files/{relative_path:path}` 对逃逸返回 404，不读取目标文件。

- [ ] **Step 1: Write the failing test**

```python
def test_file_route_rejects_escape(client, tmp_data, outside):
    outside.joinpath("secret.txt").write_text("no")
    assert client.get("/api/files/../secret.txt").status_code == 404
    assert client.get("/api/files/C:/Windows/win.ini").status_code == 404
    assert client.get("/api/files/projects\\secret.txt").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_files.py -v`  
Expected: FAIL，路由不存在，或逃逸请求没有返回 404。

- [ ] **Step 3: Implement `resolve_data_path` and the file route**

先拒绝 `\`、以 `/` 或盘符开头的路径，再解析并确认结果仍在 `data_root` 内。存在的合法文件按 PNG 或 JPEG 的媒体类型返回。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/paths.py platform/app/routers/files.py platform/app/main.py platform/tests/test_files.py
git commit -m "feat: serve only files inside the data root"
```

### Task 6: 推理服务

**Files:**
- Create: `platform/infer/segment.py`
- Create: `platform/infer/main.py`
- Create: `platform/tests/test_infer_service.py`

**Interfaces:**
- Consumes: `resolve_data_path`。官方 `segment_anything.sam_model_registry["vit_b"]`。
- Produces: `segment_image(image_path: Path, box: list[int], output_path: Path, checkpoint: Path, device: str) -> InferResult`。`InferResult` 含 `width: int`、`height: int`、`latency_ms: int`。`create_infer_app(settings: Settings) -> FastAPI`。`GET /health` 返回 `{"status": "ok"}`。

- [ ] **Step 1: Write the failing test**

```python
def test_infer_writes_mask_and_converts_grayscale(tmp_path, monkeypatch):
    seen = {}
    def fake_model(image_tensor, box):
        seen["channels"] = image_tensor.shape[1]
        return rectangle_mask(image_tensor.shape[-2], image_tensor.shape[-1], box)
    monkeypatch.setattr("platform.infer.segment.predict_mask", fake_model)
    image = tmp_path / "gray.png"
    save_grayscale(image, 8, 6)
    result = segment_image(image, [1, 1, 4, 5], tmp_path / "mask.png", tmp_path / "unused.pth", "cpu")
    mask = Image.open(tmp_path / "mask.png")
    assert seen["channels"] == 3
    assert mask.size == (8, 6) and mask.mode == "L"
    assert mask.getpixel((2, 2)) == 255 and mask.getpixel((0, 0)) == 0
    assert result.width == 8 and result.height == 6 and result.latency_ms >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_infer_service.py -v`  
Expected: FAIL，`segment_image` 不存在。

- [ ] **Step 3: Implement segmentation behind `predict_mask`**

`predict_mask(image_tensor, box)` 是唯一加载 MedSAM 的函数。二维图像先复制成三个通道。框从原图坐标线性映射到 1024，沿用 `MedSAM_Inference.py` 的预处理和阈值 0.5。测试不调用真实 `predict_mask`。另写 `test_real_demo_infer`，没有权重文件时跳过；有权重时使用 `assets/img_demo.png` 和框 `[95, 255, 190, 350]`，断言输出掩膜尺寸与原图一致且前景像素数大于零。默认命令加 `-m "not real_infer"`，该测试标记为 `real_infer`。

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_infer_service.py -m "not real_infer" -v`  
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/infer/segment.py platform/infer/main.py platform/tests/test_infer_service.py
git commit -m "feat: add MedSAM inference service"
```

### Task 7: 推理任务与 worker

**Files:**
- Create: `platform/worker/jobs.py`
- Create: `platform/worker/main.py`
- Create: `platform/app/routers/tasks.py`
- Create: `platform/app/routers/jobs.py`
- Create: `platform/tests/test_infer_jobs.py`
- Modify: `platform/app/main.py`

**Interfaces:**
- Consumes: `Task`、`Job`、`Prediction`、`InferResult`、`resolve_data_path`。
- Produces: `class InferClient(Protocol)`，方法 `infer(self, image_relative_path: str, box: list[int], output_relative_path: str) -> InferResponse`。`InferResponse` 含 `width: int`、`height: int`、`latency_ms: int`。`process_job(session: Session, job_id: UUID, client: InferClient) -> None`。`enqueue_job(connection, job_id: UUID) -> None` 把任务放进 RQ 队列 `medsam`。`POST /api/tasks/{id}/infer` 立即返回 `{ "job_id" }`，任务进入 `predicting`，job 进入 `queued`。输出相对路径是 `projects/{project_id}/predictions/{job_id}.png`。`GET /api/jobs/{id}` 成功时的 `result` 含 `prediction_id`、`mask_url`、`latency_ms`。

- [ ] **Step 1: Write the failing test**

```python
def test_box_edges_and_job_success(client, session):
    task = upload_image(client, width=10, height=8)
    edge = client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0, 0, 10, 8]})
    assert edge.status_code == 202
    assert client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0, 0, 0, 8]}).status_code == 422
    assert client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0, 0, 11, 8]}).status_code == 422
    assert client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0.5, 0, 2, 2]}).status_code == 422
    process_job(session, UUID(edge.json()["job_id"]), FakeInferClient())
    job = client.get(f"/api/jobs/{edge.json()['job_id']}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["latency_ms"] >= 0
    assert client.get(f"/api/tasks/{task['id']}").json()["status"] == "predicted"

def test_second_success_appends_prediction(client, session):
    task = upload_image(client, width=6, height=6)
    first = client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0, 0, 2, 2]})
    second = client.post(f"/api/tasks/{task['id']}/infer", json={"box": [1, 1, 3, 3]})
    process_job(session, UUID(first.json()["job_id"]), FakeInferClient())
    process_job(session, UUID(second.json()["job_id"]), FakeInferClient())
    assert session.query(Prediction).filter_by(task_id=task["id"]).count() == 2

def test_enqueue_uses_medsam_queue():
    connection = fakeredis.FakeRedis()
    job_id = UUID("00000000-0000-0000-0000-000000000001")
    enqueue_job(connection, job_id)
    assert Queue("medsam", connection=connection).job_ids == [str(job_id)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_infer_jobs.py::test_box_edges_and_job_success platform/tests/test_infer_jobs.py::test_second_success_appends_prediction platform/tests/test_infer_jobs.py::test_enqueue_uses_medsam_queue -v`  
Expected: FAIL，推理路由不存在。

- [ ] **Step 3: Implement job creation and `process_job`**

校验 `0 <= x0 < x1 <= width` 且 `0 <= y0 < y1 <= height`。测试直接调用 `process_job`，用 `FakeInferClient` 写一张全 255 掩膜。`platform/worker/main.py` 用 RQ 队列名 `medsam` 注册 `process_job`；业务服务在创建 job 后入队。模型版本使用全局约束中的字符串。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/worker/jobs.py platform/worker/main.py platform/app/routers/tasks.py platform/app/routers/jobs.py platform/app/main.py platform/tests/test_infer_jobs.py
git commit -m "feat: run MedSAM annotation jobs asynchronously"
```

### Task 8: 推理失败与重复预测

**Files:**
- Modify: `platform/worker/jobs.py`
- Modify: `platform/tests/test_infer_jobs.py`

**Interfaces:**
- Consumes: `process_job`。
- Produces: 失败时 `job.status == "failed"`，`job.error` 是异常类型和消息。已有 `Prediction` 时任务回到 `predicted`，否则回到 `pending`。再次成功会新增 `Prediction`，两条都保留。

- [ ] **Step 1: Write the failing test**

```python
def test_failure_restores_previous_prediction_without_traceback(client, session):
    task = upload_image(client, width=6, height=6)
    first = client.post(f"/api/tasks/{task['id']}/infer", json={"box": [0, 0, 2, 2]})
    process_job(session, UUID(first.json()["job_id"]), FakeInferClient())
    second = client.post(f"/api/tasks/{task['id']}/infer", json={"box": [1, 1, 3, 3]})
    process_job(session, UUID(second.json()["job_id"]), ExplodingInferClient("boom"))
    failed = client.get(f"/api/jobs/{second.json()['job_id']}").json()
    assert failed["status"] == "failed"
    assert "Traceback" not in failed["error"] and "boom" in failed["error"]
    body = client.get(f"/api/tasks/{task['id']}").json()
    assert body["status"] == "predicted"
    assert session.query(Prediction).filter_by(task_id=task["id"]).count() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_infer_jobs.py::test_failure_restores_previous_prediction_without_traceback -v`  
Expected: FAIL，失败后的任务状态或错误文本不正确。

- [ ] **Step 3: Implement the failure transition**

捕获异常后只保存 `f"{type(exc).__name__}: {exc}"`。不要把 `traceback.format_exc()` 写入数据库。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/worker/jobs.py platform/tests/test_infer_jobs.py
git commit -m "fix: preserve prior predictions when inference fails"
```

### Task 9: 人工标注

**Files:**
- Create: `platform/app/dice.py`
- Create: `platform/tests/test_annotations.py`
- Modify: `platform/app/routers/tasks.py`

**Interfaces:**
- Consumes: 最新 `Prediction` 和 `Annotation`。
- Produces: `dice(prediction: ndarray, annotation: ndarray) -> float`，按 `2 * intersection / (prediction_sum + annotation_sum)` 计算，两张空掩膜返回 `1.0`，只有一张为空返回 `0.0`。`POST /api/tasks/{id}/annotations`。`accept` 复制最新预测掩膜。`edit` 解码 `mask_png_base64`，尺寸必须与原图相同。标注相对路径是 `projects/{project_id}/annotations/{annotation_id}.png`。成功后任务为 `accepted`。

- [ ] **Step 1: Write the failing test**

```python
def test_accept_requires_prediction_and_edits_keep_history(client, session):
    task = upload_image(client, width=4, height=4)
    assert client.post(f"/api/tasks/{task['id']}/annotations", json={"action": "accept"}).status_code == 409
    edited = client.post(f"/api/tasks/{task['id']}/annotations", json={"action": "edit", "mask_png_base64": png_b64(4, 4)})
    assert edited.status_code == 201
    assert edited.json()["dice_vs_prediction"] is None
    run_prediction(client, session, task["id"], box=[0, 0, 2, 2])
    accepted = client.post(f"/api/tasks/{task['id']}/annotations", json={"action": "accept"})
    assert accepted.json()["dice_vs_prediction"] == 1.0
    assert session.query(Annotation).filter_by(task_id=task["id"]).count() == 2
    assert client.get(f"/api/tasks/{task['id']}").json()["latest_annotation"]["action"] == "accept"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_annotations.py -v`  
Expected: FAIL，标注路由不存在。

- [ ] **Step 3: Implement accept, edit, and Dice**

`accept` 在没有预测时返回 409。掩膜尺寸不符返回 422。请求可带 `annotator_name`，缺省 `annotator`。最新标注按 `created_at` 取最后一条。

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/dice.py platform/app/routers/tasks.py platform/tests/test_annotations.py
git commit -m "feat: save human annotation decisions separately from predictions"
```

### Task 10: 导出已接受任务

**Files:**
- Modify: `platform/app/routers/projects.py`
- Modify: `platform/worker/jobs.py`
- Create: `platform/tests/test_exports.py`

**Interfaces:**
- Consumes: 最新 `Annotation`、对应 `Prediction`。
- Produces: `POST /api/projects/{id}/exports` 返回 `{ "job_id" }`，job 类型为 `export`。导出目录是 `projects/{project_id}/exports/{job_id}/`，内含 `images/`、`masks/`、`predictions/` 和 `manifest.jsonl`。清单每行含 `task_id`、`filename`、`box`、`action`、`dice`、`model_version`。`Export.task_count` 等于清单行数。

- [ ] **Step 1: Write the failing test**

```python
def test_export_includes_only_accepted_tasks(client, session):
    project_id, accepted, predicted = seed_one_accepted_and_one_predicted(client, session)
    response = client.post(f"/api/projects/{project_id}/exports")
    process_job(session, UUID(response.json()["job_id"]), FakeInferClient())
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "succeeded"
    assert job["result"]["task_count"] == 1
    lines = (data_root / job["result"]["relative_path"] / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["task_id"] == accepted["id"]
    assert record["filename"] == accepted["filename"]
    assert set(record) >= {"task_id", "filename", "box", "action", "dice", "model_version"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests/test_exports.py -v`  
Expected: FAIL，导出路由不存在。

- [ ] **Step 3: Implement export inside `process_job`**

导出由 worker 执行，不调用推理客户端。没有可导出任务时，任务成功且 `task_count` 为 0。复制文件，不移动原图、预测和标注。

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest platform/tests -m "not real_infer" -v`  
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add platform/app/routers/projects.py platform/worker/jobs.py platform/tests/test_exports.py
git commit -m "feat: export only accepted annotation tasks"
```
