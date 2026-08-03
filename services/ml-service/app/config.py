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
    # m-pose 68.8, l-pose 70.4). NMS-free matters here specifically: NMS merges
    # overlapping boxes, which is how a half-visible teacher standing behind a
    # student silently disappears from a crowded frame.
    #
    # Production (RunPod L4) profile — see docs/runpod-gpu-deployment.md:
    #   DEVICE=cuda REQUIRE_DEVICE=cuda TENSORRT_EXPORT=true IMGSZ=1536
    # The engine is exported once on the pod (cached on the volume) and served
    # automatically on every later start. For a large live fleet prefer
    # yolo26l-pose (batched) over x (see docs).
    model_name: str = "auto"
    device: str = "auto"
    # Refuse to run when the resolved device is not this one (e.g. "cuda").
    # Empty disables the guard (dev boxes float between mps/cpu freely). A GPU
    # pod that silently degrades to CPU bills ~20x the wall-clock for the same
    # job, so production must die loudly instead.
    require_device: str = ""
    # On cuda: export MODEL_NAME's .pt to a TensorRT engine at first load
    # (one-time, minutes, cached next to the weight) and serve the engine.
    # fp16 engine on an L4 is the ~5x throughput lever that turns a 12-camera
    # class-day into ~3 GPU-hours. Engines are per-GPU-model, which is why the
    # export runs on the pod itself rather than in the image build.
    tensorrt_export: bool = False
    # 1280 halves a 2560px CCTV frame so small back-row people survive letterbox
    # downscale; on a GPU with headroom raise to 1536 (env IMGSZ) for more recall
    # on distant, partially occluded people. The dynamic engine export covers
    # both sizes without a rebuild.
    imgsz: int = 1280
    # 0.1 keeps the low-confidence boxes an occluded, half-visible person
    # produces; BoT-SORT's second association pass (track_low_thresh) is what
    # turns them into continued tracks instead of noise.
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
