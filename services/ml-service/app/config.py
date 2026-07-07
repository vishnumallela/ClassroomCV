"""Application settings loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgres://postgres:postgres@localhost:5433/classroom"
    model_name: str = "yolo11m-pose.pt"
    device: str = "mps"
    # Inference: 640 letterboxes the 2560-wide CCTV frame down 4x and destroys
    # back-row keypoints; 1280 is the single largest quality lever. conf=0.1
    # matches the floor the golden baseline ran with; lowering it is coupled
    # to tracker thresholds and needs offline scenario validation first.
    imgsz: int = 1280
    det_conf: float = 0.1
    max_det: int = 100
    tracker_cfg: str = str(
        Path(__file__).resolve().parent / "trackers" / "classroom_botsort.yaml"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
