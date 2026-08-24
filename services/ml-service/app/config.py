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

    # --- RF-DETR ------------------------------------------------------------
    # The one detector. A fine-tuned RF-DETR (Medium) trained on this product's
    # own five classes — door, screen, teacher, pointing, writing — which is
    # what let the whole identity stack go away: the model NAMES the teacher,
    # so nothing downstream has to infer who she is from age, behaviour or
    # appearance. Measured against the held-out room's per-frame ground truth,
    # it puts a correct box (IoU >= 0.5) on her in 95.5% of scored frames and
    # emits at most ONE teacher box per frame.
    #
    # Empty path = the service starts but /analyze fails loudly; there is no
    # second detector to silently degrade to any more, and that is deliberate.
    rfdetr_weights: str = ""
    # Inference resolution. 576 is what the checkpoint was trained at; changing
    # it re-scales every box the model learned and is not a free recall knob.
    rfdetr_resolution: int = 576
    # Frames per predict() call. RF-DETR takes a list, so batching is the main
    # GPU throughput lever; 1 on cpu/mps where it buys nothing.
    rfdetr_batch: int = 8
    # Serve the model as a TensorRT engine (cuda only, needs the `tensorrt`
    # extra). The engine is compiled for the exact GPU and TensorRT version
    # present, so it is built ON THE POD at first load and cached beside the
    # weight on the volume — never baked into the image.
    #
    # OFF by default, and the default should not change without a measurement.
    # The last time this project adopted TensorRT on a "~5x" claim, the honest
    # figure was 1.05-1.25x and the real win was an fp16 bug that had nothing
    # to do with TensorRT. Turning fp16 on (which _optimize now does on cuda)
    # is free; TensorRT costs a non-portable artifact, a heavyweight CUDA-only
    # dependency, and a post-processing path that must be parity-checked.
    rfdetr_tensorrt: bool = False

    device: str = "auto"  # auto -> cuda, else mps, else cpu
    # Refuse to run when the resolved device is not this one (e.g. "cuda").
    # Empty disables the guard (dev boxes float between mps/cpu freely). A GPU
    # pod that silently degrades to CPU bills ~20x the wall-clock for the same
    # job, so production must die loudly instead.
    require_device: str = ""

    # --- thresholds ---------------------------------------------------------
    # Set from the sweep in docs/rfdetr-pipeline.md, not by taste. The teacher
    # score is strongly bimodal on real footage (p10 = 0.79 on true positives),
    # so anything in 0.25..0.6 gives the same coverage; 0.4 sits in the middle
    # of that plateau, far from both edges.
    teacher_conf: float = 0.4
    # Door and screen are static furniture detected once per lesson, where a
    # false positive is more expensive than a miss (it would move the zone), so
    # they are held to a higher bar than the teacher.
    zone_conf: float = 0.5
    # Board-interaction classes. Reported as evidence alongside board time, so
    # they need to be right rather than plentiful.
    action_conf: float = 0.5

    # Comma-separated host[:port] allowlist for presigned media URLs. Empty
    # (default) rejects ALL URLs, so /analyze only reads local files (the SSRF
    # guard). Set to the object-store host (e.g. "minio:9000,localhost:9000")
    # to let the service fetch a video directly from MinIO/S3 by presigned URL,
    # instead of the API node downloading it to a shared filesystem.
    media_url_allowlist: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
