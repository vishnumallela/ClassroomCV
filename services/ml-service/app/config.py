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
    # Directory holding videos already present on this host, laid out like the
    # object store (<stage>/<video_id>/original.mp4). When a file is found there
    # every endpoint reads it from disk instead of fetching the object, which is
    # what makes a remote GPU worker usable over a tunnel: otherwise
    # /detect-board, /detect-door and /analyze each pull the whole video.
    media_stage_dir: str = ""
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
    # VLM verifies/overrides the GEOMETRIC teacher pick (not just the all-unknown
    # fallback): whoever the model points at (point-in-bbox) is the teacher, so a
    # student the heuristic wrongly crowned is demoted. Off by default so it is
    # A/B-testable. vlm_verify_frames is denser than vlm_frames because verify
    # runs on every video and wants coverage even when she's occluded in a frame.
    vlm_verify_teacher: bool = False
    vlm_verify_frames: int = 12
    # ...but 12 samples across an hour is one every five minutes, which is too
    # sparse to find a teacher who is often occluded, so the count grows with the
    # lesson (about one every 2.5 minutes) up to this ceiling.
    vlm_verify_frames_max: int = 26
    # Hard ceiling on vision-model calls for ONE derive. Everything downstream of
    # the teacher pick — verify, veto, trim, gap-fill — scales with how badly she
    # fragments, which grows with length, and /rederive is synchronous behind a
    # 255s transport limit. When the budget runs out the remaining calls report
    # "no answer", which every caller already treats as keep-what-you-have, so the
    # derive degrades to the geometric result instead of timing out.
    vlm_max_calls: int = 140
    # Upper bound however long the lesson is. 400 calls is roughly $0.24.
    vlm_max_calls_cap: int = 400
    # Share of the budget the per-claim stages (verify, veto, trim) may spend
    # before gap-filling gets the rest. Without this they starve it: on a
    # 37-minute lesson they consumed all 140 calls and the largest hole -- 447
    # seconds -- was never probed.
    vlm_early_stage_share: float = 0.5
    # REJECT the geometric teacher when the VLM answered at least this many frames
    # but NEVER once pointed at the geometric pick (0 point-in-bbox votes for it) and
    # found no clear teacher of its own. Distinguishes "the VLM says she isn't the
    # teacher" (demote) from "the VLM couldn't answer" (answered < this -> keep the
    # geometric pick, so a network failure never un-labels a correct teacher).
    vlm_reject_min_answered: int = 4
    # Frames sampled across a stitch CLAIM's time span to VLM-verify it before it is
    # applied. The teacher-chain stitcher can wrongly fold a student fragment that
    # ends where the teacher's tracked chain begins into the teacher (a backward
    # claim passing the height-rise test = the chimera you SEE as "student labelled
    # teacher until the real teacher comes"). If the VLM answers frames across the
    # claimed span and never points inside it, the claim is vetoed. See
    # vlm_teacher.vlm_supports_span.
    vlm_claim_verify_frames: int = 6
    # VLM entry-backfill (reclaim the teacher's pre-chain entry/seated detections).
    # OFF by default: on real crowded footage her entry is hyper-fragmented and
    # occluded by foreground students, so the trajectory trace over-claims (grabs
    # students). Kept behind a flag for future work — a robust version needs a
    # per-candidate-fragment VLM verification pass (see vlm_teacher.vlm_supports_span),
    # which is many more VLM calls. See vlm_teacher.backfill_teacher_entry.
    vlm_backfill_entry: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
