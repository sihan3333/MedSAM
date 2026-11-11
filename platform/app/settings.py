from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = "postgresql+psycopg://medsam:medsam@127.0.0.1:5432/medsam"
    data_root: Path = Path("platform/data")
    medsam_checkpoint: Path = Path("work_dir/MedSAM/medsam_vit_b.pth")
    infer_url: str = "http://127.0.0.1:8001"
    device: str = "cpu"
    redis_url: str = "redis://127.0.0.1:6379/0"
