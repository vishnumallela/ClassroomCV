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

## 2. RunPod pod — entirely from the Settings page

There is nothing to do in the RunPod console. The app holds the whole pod
specification and provisions against RunPod's API, so the only thing you paste
in is an API key.

1. `deploy-ml-runpod.yml` pushes the image to `ghcr.io/<owner>/<repo>/ml-service`
   (make the package public, or add registry credentials on RunPod).

   **Mind the tag.** Only `main` may write `:latest`; a branch publishes under
   its own sanitized name. While the RF-DETR rewrite sits on
   `feat/rfdetr-pipeline`, `:latest` is still the *previous* YOLO/identity-stack
   service — same repository, same port, same `/health`, a different program.
   A pod on it starts, reports RUNNING, silently ignores every `RFDETR_*`
   variable (the settings model is `extra="ignore"`), and has no entrypoint and
   therefore no sshd to diagnose it with, while the GPU bills. The default in
   `POD_DEFAULTS` is the branch tag for exactly this reason; **switch it back to
   `:latest` when this branch merges.**

   You do not have to remember this: the Settings page reads the configured
   tag's own config out of the registry and reports which service it contains,
   and refuses to create a pod on an image that is not the ML service or on a
   tag that does not exist (`lib/registry.ts`).
2. Open **Settings** and paste a **RunPod API key** (read/write scope). Every
   dropdown below then fills from RunPod's live catalog.
3. **Machine** — pick the GPU (each option carries its real current $/hr and
   only appears if it is actually purchasable on the selected tier), the cloud
   tier, and the region. Regions that cannot hold a network volume are not
   offered, because the checkpoint has to live in the pod's own region.
4. **Network volume** — select an existing one or create it inline. It holds
   the RF-DETR checkpoint and the video scratch, and it outlives every pod.
5. **Image and environment** — image tag, container disk, allowed CUDA
   versions, and the ml-service env (`RFDETR_*`, `MEDIA_URL_ALLOWLIST`,
   `DATABASE_URL`). `MEDIA_URL_ALLOWLIST` must be the MinIO public host and
   `DATABASE_URL` the Railway Timescale service's public URL; the page warns
   when either is missing rather than letting you create a pod that cannot work.
6. **Create pod.** The page then shows its state, GPU, region and hourly rate,
   and whether `/health` is answering.

Populating a **fresh** volume with the checkpoint is the one step that needs a
shell: add your SSH public key in Settings before creating the pod (the image
runs sshd for exactly this), then rsync the `.pth` to
`/workspace/weights/`. Once it is there, no later pod needs this.

`allowedCudaVersions` defaults to `13.0,12.8` and should stay pinned: `uv.lock`
resolves torch 2.12.1 to a cu13 wheel needing driver r580+, and an unpinned pod
can land on a 12.4 host, come up "healthy", and run the whole batch on CPU.

## 3. Autopilot

- **Auto-provision GPU** — a queued lesson with no GPU serving it creates a pod
  from this spec (or starts a stopped one). An account with no pod at all
  recovers on its own; nobody has to open a browser.
- **Idle release** — once the queue has been empty for the configured minutes,
  the GPU is released. **Terminate** is the default and the right choice: a
  *stopped* pod stays pinned to its host machine while that machine's GPU is
  re-rented, and the restart then fails with "not enough free GPUs on the host
  machine". Billing ends either way, and the checkpoint is on the volume, so
  the replacement pod comes up ready.

Together these make the rent-by-the-hour model hands-off: cost floor is the
volume (~$0.07/GB/month), and GPU time is bought only while lessons are running.

Lessons uploaded while the GPU is down simply queue (BullMQ delays without
consuming retries) and process when a pod is serving again.

**ML service URL** should stay empty. With a pod, the URL is derived from its
id — RunPod's proxy hostname is the one address that survives a pod's whole
life, so autopilot never leaves a stale URL behind. Fill it in only to override
(a tunnel, a second pod); it applies immediately, no redeploy.

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
- No per-lesson API cost. One detector runs on the pod and nothing calls out;
  the small hard-capped Gemini vote that used to break ties on who the teacher
  was went away with the rest of the identity stack.
- Storage: MinIO on a Railway volume for media, TimescaleDB rows slimmed
  to the three teacher KPIs; only the teacher's raw detections are stored at
  all, and they age out per the retention policy.
