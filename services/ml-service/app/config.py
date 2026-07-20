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

    # Model + device are device-aware and env-overridable, so the SAME code runs
    # on a Mac (MPS) for development and on an NVIDIA GPU in production.
    #
    #   device="auto"     -> cuda if available, else mps, else cpu.
    #   model_name="auto" -> the best YOLO26 pose model for the resolved device:
    #                        yolo26x-pose on cuda (a GPU has the headroom),
    #                        yolo26m-pose on mps/cpu (keeps dev iteration fast).
    #
    # YOLO26 is NMS-free and reports up to +7.2 pose AP over YOLO11 (COCO
    # m-pose 68.8, l-pose 70.4). On a GPU, export to TensorRT for a ~5x fp16
    # speedup and point MODEL_NAME at the engine, e.g.:
    #   yolo export model=yolo26x-pose.pt format=engine half=True dynamic=True
    #   MODEL_NAME=yolo26x-pose.engine DEVICE=cuda IMGSZ=1536
    # For a large live fleet prefer yolo26l-pose (batched) over x (see docs).
    model_name: str = "auto"
    device: str = "auto"
    # 1280 halves a 2560px CCTV frame so small back-row people survive letterbox
    # downscale; on a GPU with headroom raise to 1536 (env IMGSZ) for more recall.
    imgsz: int = 1280
    det_conf: float = 0.1
    max_det: int = 100
    # Comma-separated host[:port] allowlist for presigned media URLs. Empty
    # (default) rejects ALL URLs, so /analyze only reads local files (the SSRF
    # guard). Set to the object-store host (e.g. "minio:9000,localhost:9000")
    # to let the service fetch a video directly from MinIO/S3 by presigned URL,
    # instead of the API node downloading it to a shared filesystem.
    media_url_allowlist: str = ""
    tracker_cfg: str = str(
        Path(__file__).resolve().parent / "trackers" / "classroom_botsort.yaml"
    )

    # Vision-LLM teacher-ID fallback (Gemini native generateContent API). Runs ONLY
    # when the geometric ranker returns all-unknown: a teacher who comes in and
    # sits the whole lesson scores like a student (low standing/movement), so pose
    # and motion cannot find her, but a vision model can (adult, not in uniform, at
    # the teacher's desk). Empty GEMINI_API_KEY disables it, so most videos never
    # touch the API. See app/vlm_teacher.py.
    #
    # vlm_model defaults to the "-latest" alias, not a dated version: Gemini 2.5
    # Flash/Flash-Lite already 404 for new API keys despite still being listed in
    # the model catalog, so a pinned version silently breaks on the next
    # deprecation. The alias keeps pointing at Google's current stable Flash model.
    gemini_api_key: str = ""
    vlm_model: str = "gemini-flash-latest"
    vlm_teacher_fallback: bool = True
    vlm_frames: int = 6  # frames sampled across the lesson to vote over
    vlm_min_votes: int = 2  # the winning track must win at least this many frames


@lru_cache
def get_settings() -> Settings:
    return Settings()
