# Deployment: Railway (app) + RunPod (GPU) with automatic CI/CD

The application lives on **Railway**; the ML pipeline runs **only on a RunPod
GPU pod** — on-demand, never serverless, so it can be stopped from the app's
Settings page and the meter stops with it. CI/CD is automatic on push to
`main`.

```
GitHub main ──► Railway build (api, web)   ── app code deploys itself
        ├────► deploy-railway.yml ─────────► railway config apply (infrastructure)
        └────► deploy-ml-runpod.yml ───────► GHCR image ─► RunPod pod restart (pulls :latest)

browser ─► web (Caddy: SPA + /api proxy) ─► api ─► Timescale / Redis / MinIO
                                              └──► RunPod pod :8000 (L4 GPU)
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
| `web` | this repo, `apps/frontend/Dockerfile` | Vite SPA built at image-build time, served by Caddy with an SPA fallback. Also proxies `/api` to `api` — the browser only ever talks to this one origin |

Both Dockerfiles build from the **repo root** (Bun workspaces need the root
lockfile), which is why each service sets `build.dockerfilePath` rather than a
root directory.

Two things the config file deliberately does not contain:

- **Public domains.** They are generated per environment
  (`railway domain --service api`) and referenced as
  `${{RAILWAY_PUBLIC_DOMAIN}}`, so the wiring follows whatever Railway hands
  out. Because Vite inlines `FRONTEND__API_URL` at **build** time, a domain
  must exist before `web` builds — generate domains first, then redeploy.
- **Secrets.** They are platform-managed and declared `preserve()`. Do **not**
  put a `generator:` on a live credential — it re-runs on every
  `config apply`, so an unrelated infrastructure change silently rotates it.
  That rotated `POSTGRES_PASSWORD` after initdb had already baked the old one
  into the data directory, locking the api out of its own database. Set them
  once instead, and the file will never touch them again:

  ```bash
  railway variables set POSTGRES_PASSWORD=…                     --service Timescale
  railway variables set MINIO_ROOT_PASSWORD=…                   --service minio
  railway variables set API_SERVICE__ADMIN_PASSWORD=…           --service api
  railway variables set API_SERVICE__QUEUE_DASHBOARD_PASSWORD=… --service api
  ```

  Read one back with `railway variables list --service api --kv`. Setting a
  secret with `--skip-deploys` leaves the running container on the old value —
  `railway redeploy --service <name>` afterwards, or MinIO answers
  `SignatureDoesNotMatch` to every S3 call.

**The SPA and the API must share an origin.** Caddy proxies `/api` on the web
service through to `api` for exactly one reason: the admin session is a
cookie, and browsers discard a `Set-Cookie` that arrives from a different
origin. Point the SPA at the api's own domain and `/auth/login` answers 200
while the session never sticks — the lock screen simply never opens. Same
origin also means no CORS preflight on a multi-GB upload, and `<video>`
elements send the session without a `crossorigin` dance.

Three platform behaviours worth knowing before changing this file:

- **MinIO must not run its console.** Left on, it opens a *random* high port,
  which gives Railway's proxy a second listener to choose from and it answers
  502 on the S3 API. `MINIO_BROWSER=off` plus an explicit `PORT` leaves one.
  It also needs `--address [::]:9000`; the bare `:9000` form is IPv4-only and
  Railway routes over IPv6.
- **`database(name, "postgres", { image })` cannot deliver TimescaleDB.** It
  provisions Railway's own Postgres image first and swaps the custom one in
  afterwards — initdb has already run, so the cluster ends up the wrong major
  version with no `timescaledb` in `shared_preload_libraries`. Declare the
  service straight from the image, as this file does.
- **Never delete a volume out of band.** Doing so leaves the service attached
  to a pending-deletion volume, which `config plan` cannot read through — so
  it plans another create, Railway refuses the second volume, and every apply
  re-enters the loop. `volume detach` does not clear it and the pending
  deletion cannot be cancelled; the only exit is deleting the service and
  letting the config recreate it against a differently-named volume.

No ML host is pinned here: the api resolves the ML service URL from app
settings at call time (§3), so re-pointing at a fresh pod needs no redeploy.

### Live deployment

Project `classroomcv` in the **Ravi Sankar's Projects** workspace.

| | |
|---|---|
| App | <https://web-production-cc3f0e.up.railway.app> — also serves the API at `/api` |
| API (direct) | <https://api-production-5260.up.railway.app> (`/health`, queue dashboard at `/admin/queues`) |
| Object storage | <https://minio-production-14a9.up.railway.app> (S3 API only, credentialed) |
| Database | `timescale.railway.internal:5432` in-network; TCP proxy for the GPU pod |

`railway config plan` reports one standing `Update Timescale networking` diff
it cannot converge — the CLI does not read the TCP proxy back, so it re-plans
the same no-op every time. Harmless; the proxy is live.

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
