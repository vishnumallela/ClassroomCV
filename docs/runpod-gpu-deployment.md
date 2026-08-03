# Running ml-service on RunPod (GPU, TensorRT)

The ML pipeline runs **only on the RunPod GPU worker** in production. The
Azure VM hosts frontend / api-service / Postgres / MinIO; the pod pulls video
by presigned URL, runs the pipeline, and posts results back. Nothing here is
live — one 12-camera class-day is ~3 GPU-hours on an L4, so the pod is started
for the batch and stopped after.

## Pod specification

| Item | Value | Why |
|---|---|---|
| GPU | **NVIDIA L4 24 GB** (on-demand ~$0.39/h; spot is safe — batch jobs resume) | Video analytics is decode+throughput bound; L4 has strong NVDEC. H100 is 5–8x the price for no useful gain. |
| CPU / RAM | 8 vCPU / 32 GB | ffmpeg/cv2 decode feeds the GPU; starving the CPU strands the card. |
| Volume | 100 GB at `/workspace` | Model weight + TensorRT engine cache + per-job video scratch. |
| Image | `services/ml-service/Dockerfile` (PyTorch 2.7 + CUDA 12.6 runtime) | — |

## Environment

Copy `services/ml-service/.env.runpod.example`. The load-bearing settings:

- `DEVICE=cuda REQUIRE_DEVICE=cuda` — the pod **refuses to run** if CUDA
  didn't resolve. A silent CPU fallback bills ~20x the wall-clock, which is
  the whole GPU budget gone on one job.
- `TENSORRT_EXPORT=true` — at first model load the service exports
  `yolo26x-pose.pt` to a **fp16 TensorRT engine** (one-time, minutes, cached
  on the volume) and serves the engine from then on (~5x throughput).
  Engines are per-GPU-model and per-TensorRT-version: after changing either,
  delete the `.engine` and restart (or run
  `python scripts/export_tensorrt.py`). Engines never fall back to CPU —
  a broken engine fails the job loudly instead of silently crawling.
- `IMGSZ=1536` — recall on small back-row and half-occluded people. The
  engine is exported `dynamic=True`, so 1280 and 1536 both run without a
  rebuild.
- `MEDIA_URL_ALLOWLIST=<storage-host>` — the SSRF gate; only the object
  store may be fetched. Downloads are chunked, capped, and **resume from the
  last byte written** when the WAN transfer stalls.

## Occlusion posture (why these settings)

Classroom footage fails by **fragmentation**, not ID switches: the teacher
crosses behind rows of students half-visible and her track dies and is
reborn. The pipeline is set up for that end to end:

- YOLO26 is **NMS-free** — NMS is what merges a half-visible teacher into the
  student box in front of her; set prediction keeps both.
- `det_conf=0.1` + BoT-SORT `track_low_thresh=0.1` — the ByteTrack second
  pass: occluded people score low but their boxes are honest, and the second
  association carries them through the crowd.
- `track_buffer=60` — a lost track survives ~12 s of occlusion at the 5 fps
  sampling cap before deletion (was ~6 s, measured too short for a teacher
  crossing behind a front row).

## Cost

From `docs/infrastructure-provisioning-classroomcv-hiringai.md` (Aug 2026):
~2.9 GPU-hours per 12-camera class-day ≈ **$1.15/day** on-demand, ~$25/month
per classroom; budget 2x for pod start + model load + transfer, and use spot
for a further 30–50%. There are **no per-frame API costs in this pipeline** —
everything (YOLO26 pose, BoT-SORT, CLIP re-ID, YOLOE zones) runs on the pod;
keep it that way: a vision-LLM call per sampled frame would dwarf the GPU
line item.

## Bring-up checklist

1. Build + push the image; create the pod with the volume at `/workspace`.
2. Set env from `.env.runpod.example` (storage host, DATABASE_URL).
3. First start: watch logs for `TensorRT export complete` then
   `warmup inference complete on cuda`.
4. `GET /health` → `{"device": "cuda", "model": ".../yolo26x-pose.engine"}`.
5. Submit one short clip via `/analyze` end-to-end before the first real batch.
