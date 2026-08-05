# Deployment: Railway (app) + RunPod (GPU) with automatic CI/CD

The application lives on **Railway**; the ML pipeline runs **only on a RunPod
GPU pod** — on-demand, never serverless, so it can be stopped from the app's
Settings page and the meter stops with it. CI/CD is automatic on push to
`main`.

```
GitHub main ──► Railway build (api, web)   ── app code deploys itself
        ├────► deploy-railway.yml ─────────► railway config apply (infrastructure)
        └────► deploy-ml-runpod.yml ───────► GHCR image ─► RunPod pod restart (pulls :latest)

browser ─► web (Caddy/SPA) ─► api ─► Timescale / Redis / MinIO
                                └──► RunPod pod :8000 (ml-service, L4 GPU)
```

## 1. Railway project

The whole project is **declared in code** at [`.railway/railway.ts`](../.railway/railway.ts)
— services, volumes, variables and the wiring between them. There is no
click-through setup to reproduce:

```bash
railway link                # pick the classroomcv project
railway config plan         # preview (safe)
railway config apply        # reconcile
```

| Resource | What it is | Why |
|---|---|---|
| `Timescale` | `timescale/timescaledb:latest-pg17` + volume | Railway's managed Postgres has no TimescaleDB, and the schema needs hypertables, compression, retention and a continuous aggregate |
| `Redis` | Railway Redis | BullMQ broker for the upload → analyse → derive pipeline |
| `minio` | `minio/minio:latest` + volume | S3-compatible store for video + thumbnail bytes. Also what lets the *remote* GPU worker read a lesson: it gets a presigned URL, never a path on the api's disk |
| `api` | this repo, `apps/api-service/Dockerfile` | Bun + Hono + the BullMQ workers. Migrations run as a pre-deploy step; healthcheck `/health` |
| `web` | this repo, `apps/frontend/Dockerfile` | Vite SPA built at image-build time, served by Caddy with an SPA fallback |

Both Dockerfiles build from the **repo root** (Bun workspaces need the root
lockfile), which is why each service sets `build.dockerfilePath` rather than a
root directory.

Two things the config file deliberately does not contain:

- **Public domains.** They are generated per environment
  (`railway domain --service api`), and referenced as
  `${{api.RAILWAY_PUBLIC_DOMAIN}}` / `${{web.RAILWAY_PUBLIC_DOMAIN}}` so the
  CORS origin and the SPA's API origin follow whatever Railway hands out.
  Because Vite inlines `FRONTEND__API_URL` at **build** time, the api's domain
  must exist before `web` builds — generate domains first, then redeploy `web`.
- **Secrets.** `API_SERVICE__ADMIN_PASSWORD`, the queue-dashboard password and
  the MinIO root password are declared as Railway `secret(…)` generators, so
  they are created on the platform and never live in the repo. Read the admin
  password back with `railway variables list --service api --kv`.

No ML host is pinned here: the api resolves the ML service URL from app
settings at call time (§3), so re-pointing at a fresh pod needs no redeploy.

## 2. RunPod pod (one-time)

1. `deploy-ml-runpod.yml` pushes the image to `ghcr.io/<owner>/<repo>/ml-service`
   (make the package public, or add registry credentials on RunPod).
2. Create an **on-demand GPU pod** (NOT serverless): L4 24 GB, 8 vCPU/32 GB,
   image `ghcr.io/…/ml-service:latest`, expose HTTP port **8000**, volume
   **100 GB** at `/workspace` (holds the weight + TensorRT engine across
   restarts). Env: see `services/ml-service/.env.runpod.example` —
   `MEDIA_URL_ALLOWLIST` must be the MinIO public host, `DATABASE_URL` the
   Railway Timescale service's public URL.
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

Application code deploys **itself**: `api` and `web` are connected to this
repo, so Railway builds and releases each push to `main` that matches their
watch patterns. Nothing in CI runs `railway up`.

| Pipeline | Trigger | What it does |
|---|---|---|
| Railway GitHub integration | push to `main` under `apps/api-service/**`, `packages/**`, `package.json`, `bun.lock` | builds and releases `api` (migrations run pre-deploy) |
| Railway GitHub integration | push to `main` under `apps/frontend/**`, `packages/**`, `package.json`, `bun.lock` | builds and releases `web` |
| `.github/workflows/ci.yml` | every push and PR | typecheck + build the workspaces, `pytest` the ML service |
| `.github/workflows/deploy-railway.yml` | changes under `.railway/**` | `railway config plan` then `apply` — keeps the live project from drifting from the committed infrastructure |
| `.github/workflows/deploy-ml-runpod.yml` | changes under `services/ml-service/**` | builds + pushes the GHCR image, then stop→start on the pod so it pulls `:latest` |

**Repo configuration:**

- Secrets: `RAILWAY_TOKEN` (a project token for `classroomcv`),
  `RUNPOD_API_KEY`, `RUNPOD_POD_ID`
- Variables: `RAILWAY_DEPLOY_ENABLED=true`, `RUNPOD_DEPLOY_ENABLED=true`

Leave the `*_DEPLOY_ENABLED` variables unset to keep a workflow inert — the
pipeline never fails because deploy credentials are absent. Code deploys are
unaffected either way; they come from Railway's own GitHub integration.

## 5. Cost posture

- GPU: start for the batch, stop after — ~3 GPU-hours per 12-camera class-day
  (~$1.15 on-demand). The Settings page's Stop button is the lever.
- Gemini: hard-capped at `vlm_frames` (6) calls per lesson for teacher ID —
  ~$0.002/lesson, zero when `GEMINI_API_KEY` is unset.
- Storage: MinIO on a Railway volume for media, TimescaleDB rows slimmed
  to the three teacher KPIs; raw detections age out per the retention policy.
