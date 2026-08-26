# Running ml-service on RunPod (GPU)

The ML pipeline runs **only on the RunPod GPU worker** in production. The
Azure VM hosts frontend / api-service / Postgres / MinIO; the pod pulls video
by presigned URL, runs the pipeline, and posts results back. Nothing here is
live — one 12-camera class-day is a few GPU-hours on an L4, so the pod is
provisioned for the batch and destroyed after.

**The pod is created and destroyed from the app's Settings page, not the RunPod
console.** The app holds the whole specification below and provisions against
RunPod's API; see `docs/deployment.md` for the walkthrough. This file explains
*why* the values are what they are, which is what you need when changing one.

## Pod specification

| Item | Value | Why |
|---|---|---|
| GPU | **NVIDIA L4 24 GB** (on-demand ~$0.49/h; spot is safe — batch jobs resume) | Video analytics is decode+throughput bound; L4 has strong NVDEC. H100 is 5–8x the price for no useful gain. |
| CPU / RAM | 8 vCPU / 32 GB | ffmpeg/cv2 decode feeds the GPU; starving the CPU strands the card. |
| Volume | 100 GB at `/workspace` | RF-DETR checkpoint + per-job video scratch. Outlives every pod — that is what makes terminate-on-idle cheap. |
| CUDA | pinned to **13.0 / 12.8** at create time | `uv.lock` resolves torch 2.12.1 to a cu13 wheel needing driver r580+. Unpinned, RunPod may place the pod on a 12.4 host; it boots, reports healthy, and runs on CPU at ~20x the wall-clock. |
| Image | `services/ml-service/Dockerfile` (PyTorch 2.12.1 + CUDA 13.0 runtime) | Must match `uv.lock`; see the Dockerfile header. |
| Image **tag** | `:feat-rfdetr-pipeline` until this branch merges, then `:latest` | Only `main` writes `:latest`, so today it is still the old YOLO service — identical name and `/health`, different program, no entrypoint, and it ignores every `RFDETR_*` var. Settings verifies the tag against the registry and blocks a wrong one. |

## Environment

Set on the Settings page, which passes it to the container at create time;
`services/ml-service/.env.runpod.example` is the same list for a hand-run pod.
The load-bearing settings:

- `DEVICE=cuda REQUIRE_DEVICE=cuda` — the pod **refuses to run** if CUDA
  didn't resolve. A silent CPU fallback bills ~20x the wall-clock, which is
  the whole GPU budget gone on one job.
- `RFDETR_WEIGHTS=/workspace/weights/…` — the fine-tuned checkpoint, on the
  **volume**, not in the image: it is a ~260 MB artifact on its own retraining
  cadence, and the container layer is recreated on every pod stop/start. There
  is no fallback detector, so an unset or missing path fails `/analyze` loudly
  rather than degrading to something else.
- `RFDETR_BATCH=16` — frames per `predict()` call. RF-DETR takes a list, so
  this is the main throughput lever on a GPU; raise while VRAM allows.
- `MEDIA_URL_ALLOWLIST=<storage-host>` — the SSRF gate; only the object
  store may be fetched. Downloads are chunked, capped, and **resume from the
  last byte written** when the WAN transfer stalls.

`RFDETR_RESOLUTION` (default 576) is the resolution the checkpoint was trained
at. It is not a recall knob: changing it rescales every box the model learned.

## GPU precision (do this first — it is free)

On cuda the model is JIT-traced in **fp16 at `RFDETR_BATCH`**. Both of rfdetr's
own defaults are wrong for a pod: `dtype` defaults to `float32`, and
`batch_size` defaults to 1 while we feed batches of 16, so an untuned load runs
half-speed at double the VRAM on a graph specialised for the wrong shape. This
is handled in `detector._optimize` and pinned by tests.

It is called out because this project has already lost time to exactly this:
the previous pipeline's celebrated "~5x TensorRT speedup" measured 1.05–1.25x
once someone checked, and the real win was a warmup call that had been silently
pinning the backend to fp32. **fp16 is most of the win and costs nothing.**

## TensorRT

Enable with `RFDETR_TENSORRT=true` plus `uv sync --extra tensorrt`. The engine
is built on the pod at first load and cached next to the weight as
`<name>.r<resolution>.trt`.

Three things to know before you turn it on:

1. **The engine is not portable.** TensorRT compiles for the exact GPU model
   and TensorRT version present at build time. It is therefore built on the pod
   and cached on the volume, never baked into the image. After changing GPU
   type, delete the `.trt` file and let it rebuild.
2. **rfdetr ships TensorRT export but not TensorRT inference.** Its own
   `_tensorrt` module points at a separate library for serving. So
   `app/tensorrt_backend.py` supplies the missing half — preprocessing and
   batching around rfdetr's *own* `PostProcess`, with `means`/`stds` read off
   the loaded model rather than hardcoded, so decoding cannot drift. What is
   ours is still ours, and a mismatch there would not raise: it would shift
   every box and every KPI.
3. **Therefore: parity-check before trusting it.**

```bash
# On the pod, against a real lesson:
uv run --extra tensorrt python tools/trt_parity.py /workspace/data/lesson.mp4
```

That compares teacher-class detections between the two backends at the
production threshold, and prints the real speedup against **fp16** PyTorch —
not against the fp32 strawman that produced the last inflated number. If the
ratio is near 1.0, leave `RFDETR_TENSORRT=false`: a non-portable artifact and a
CUDA-only dependency are a poor trade for nothing.

`GET /health` reports `"backend": "tensorrt" | "pytorch"` — read back from what
actually loaded, because "enabled" and "running" are different claims. Every
failure in the TensorRT path (missing extra, failed export, failed load)
degrades to fp16 PyTorch and logs why.

## Cost

~2.9 GPU-hours per 12-camera class-day ≈ **$1.15/day** on-demand, ~$25/month
per classroom; budget 2x for pod start + model load + transfer, and use spot
for a further 30–50%. There are **no per-frame API costs in this pipeline** —
one detector runs on the pod and nothing calls out. Keep it that way: a
vision-LLM call per sampled frame would dwarf the GPU line item. (The previous
pipeline did make a small, hard-capped vision-model call per *lesson* to break
ties on who the teacher was; a trained `teacher` class removed the need.)

## Bring-up checklist

All of steps 1–4 are the Settings page; none of them is the RunPod console.

1. Build + push the image (CI does this on every push to `services/ml-service/**`).
2. Settings → paste the RunPod API key, pick the GPU and region, select or
   create the network volume, set `MEDIA_URL_ALLOWLIST` and `DATABASE_URL`.
   The page names anything still missing.
3. **Create pod.**
4. Watch the status card: it reports the pod's state and hourly rate, then
   `/health` once uvicorn answers. Expect
   `{"device": "cuda", "model": "/workspace/weights/…"}` — `device: cpu` there
   means the CUDA pin did not hold, and the pod should be terminated, not used.
5. Pod logs on first start: `loading RF-DETR … on device cuda` and `RF-DETR
   optimized for inference`. A checkpoint whose class order disagrees with
   `app/models.py` refuses to load here rather than silently reporting the door
   as the teacher.
6. Submit one short clip via `/analyze` end-to-end before the first real batch.

## Running against a laptop (MinIO + Postgres are local)

In production the pod reaches object storage and the database over the public
internet. Against a dev laptop it cannot, and **two of the three legs of an
analysis run inbound**, so they need reverse tunnels:

```bash
services/ml-service/tools/pod_tunnel.sh
```

It reads the live pod's IP and SSH port out of the running api-service — a new
pod gets both fresh every time, so a pasted-in endpoint tunnels to nothing —
and holds `pod:9000 -> MinIO` and `pod:5533 -> Postgres` open in a reconnect
loop. Leave it running for the whole analysis.

Do **not** rewrite the storage host to make it reachable: the presigned URL's
host is what SigV4 signs *and* what `MEDIA_URL_ALLOWLIST` checks, so it has to
resolve to the same name inside the pod. That is why the tunnel maps port 9000
to port 9000.

Without it, analysis fails with `<urlopen error [Errno 111] Connection
refused>` — which reads like a broken GPU and is only a missing route. The
third leg, api-service to pod, needs nothing: it goes out over RunPod's proxy
hostname.

**Once per volume, not per pod:** a brand-new volume has no checkpoint on it, so
`/analyze` fails with "checkpoint not found" however healthy the pod looks. Add
your SSH public key in Settings *before* creating the pod, then rsync the `.pth`
to `/workspace/weights/` using RunPod's "SSH over exposed TCP" line — the
`ssh.runpod.io` proxy does not support SCP/SFTP. `tools/pod_setup.sh` does this
and more for a hand-built pod.
