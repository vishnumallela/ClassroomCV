# Deployment: Railway (app) + RunPod (GPU) with automatic CI/CD

The application lives on **Railway**; the ML pipeline runs **only on a RunPod
GPU pod** — on-demand, never serverless, so it can be stopped from the app's
Settings page and the meter stops with it. CI/CD is automatic on push to
`main`.

```
GitHub main ──► deploy-railway.yml ──► Railway: frontend + api-service
        └────► deploy-ml-runpod.yml ─► GHCR image ─► RunPod pod restart (pulls :latest)

browser ─► frontend ─► api-service ─► Postgres(Timescale) / Redis / R2
                              └──────► RunPod pod :8000 (ml-service, L4 GPU)
```

## 1. Railway project (one-time)

Create a Railway project with these services:

| Service | Source | Notes |
|---|---|---|
| `api-service` | this repo, root dir `apps/api-service` | config in `apps/api-service/railway.json` (migrates on boot, healthcheck `/health`) |
| `frontend` | this repo, root dir `apps/frontend` | config in `apps/frontend/railway.json` |
| `db` | Docker image `timescale/timescaledb:latest-pg16` + volume | Railway's managed Postgres lacks TimescaleDB Community (compression/retention), so run the image |
| `redis` | Railway Redis | BullMQ queue |

Media bytes go to **Cloudflare R2** (zero egress — the RunPod pod pulls video
from it): create a bucket + API token.

**api-service variables**

```
API_SERVICE__ADMIN_PASSWORD=…               # REQUIRED in production: gates the whole app
API_SERVICE__DATABASE_URL=postgres://…      # the db service
API_SERVICE__REDIS_URL=redis://…
API_SERVICE__CORS_ORIGINS=https://<frontend-domain>
API_SERVICE__STORAGE_BACKEND=s3
API_SERVICE__S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
API_SERVICE__S3_BUCKET=luminary-videos
API_SERVICE__S3_ACCESS_KEY=…  API_SERVICE__S3_SECRET_KEY=…
API_SERVICE__DATA_DIR=/tmp/luminary-data    # ephemeral cache; S3 is the source of truth
```

**frontend variables** (Vite inlines env at BUILD time — set before deploying)

```
FRONTEND__API_URL=https://<api-service-domain>
```

## 2. RunPod pod (one-time)

1. `deploy-ml-runpod.yml` pushes the image to `ghcr.io/<owner>/<repo>/ml-service`
   (make the package public, or add registry credentials on RunPod).
2. Create an **on-demand GPU pod** (NOT serverless): L4 24 GB, 8 vCPU/32 GB,
   image `ghcr.io/…/ml-service:latest`, expose HTTP port **8000**, volume
   **100 GB** at `/workspace` (holds the weight + TensorRT engine across
   restarts). Env: see `services/ml-service/.env.runpod.example` —
   `MEDIA_URL_ALLOWLIST` must be the R2 host, `DATABASE_URL` the Railway db.
3. First boot exports the TensorRT engine (minutes, once per GPU type); watch
   for `warmup inference complete on cuda`.

## 3. Wire it in the app (Settings page)

Open **Settings** in the app and fill in:

- **RunPod API key + pod ID** — enables the Start/Stop GPU buttons and the
  live status card. Stop when the batch is done; a stopped pod bills volume
  storage only (~cents/month) instead of ~$0.39/hr.
- **GPU autopilot** — "Auto-start GPU" boots the pod when a lesson is queued
  while it's off; "Auto-stop after idle" shuts it down once the queue has
  been empty for the configured minutes. Together they make the
  rent-by-the-hour model fully hands-off.
- **ML service URL** — the pod's exposed port, e.g.
  `https://<podId>-8000.proxy.runpod.net`. Applies immediately (no redeploy);
  it overrides the env default.

Lessons uploaded while the GPU is off simply queue (BullMQ retries with
backoff) and process when the pod starts.

## 4. CI/CD (automatic on push to main)

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/deploy-railway.yml` | changes under `apps/**`, `packages/**` | `railway up` for api-service and frontend |
| `.github/workflows/deploy-ml-runpod.yml` | changes under `services/ml-service/**` | builds + pushes the GHCR image, then stop→start on the pod so it pulls `:latest` |

**Repo configuration:**

- Secrets: `RAILWAY_TOKEN`, `RUNPOD_API_KEY`, `RUNPOD_POD_ID`
- Variables: `RAILWAY_DEPLOY_ENABLED=true`, `RAILWAY_API_SERVICE`,
  `RAILWAY_FRONTEND`, `RUNPOD_DEPLOY_ENABLED=true`

Leave the `*_DEPLOY_ENABLED` variables unset to keep a workflow build-only —
the pipeline never fails because deploy credentials are absent.

## 5. Cost posture

- GPU: start for the batch, stop after — ~3 GPU-hours per 12-camera class-day
  (~$1.15 on-demand). The Settings page's Stop button is the lever.
- Gemini: hard-capped at `vlm_frames` (6) calls per lesson for teacher ID —
  ~$0.002/lesson, zero when `GEMINI_API_KEY` is unset.
- Storage: R2 for media (no egress fees to RunPod), TimescaleDB rows slimmed
  to the three teacher KPIs; raw detections age out per the retention policy.
