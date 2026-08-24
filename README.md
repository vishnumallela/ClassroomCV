# Luminary

_Every lesson, brought to light._

Register a classroom, configure its board and door zones once, and upload
lessons to get trustworthy teaching analytics — exactly three teacher KPIs:
**entries/exits, time at the board, and a movement heatmap** — each shown with
an honest confidence level, never a falsely precise number. No facial
recognition, no named students, aggregate teacher insights only.

A single fine-tuned **RF-DETR** does the detection — five classes (`door`,
`screen`, `teacher`, `pointing`, `writing`) — running on an on-demand RunPod GPU
controlled from the app's Settings page; the dashboard reads the results over a
typed API. Because the model names the teacher directly, there is no person
re-identification, no multi-object tracker and no vision-model tiebreak: what
follows detection is a plausible-motion check and a gap bridge
(`services/ml-service/app/teacher.py`). **Students are never detected, never
stored and never drawn** — only the teacher's boxes reach the database.

Every analysed lesson carries a **data-quality report**
(`services/ml-service/app/quality.py`) — how much of the lesson she was visible
for, how often her timeline broke, and how confident the detector was — so the
dashboard can say how much each figure can be trusted.

[`docs/rfdetr-pipeline.md`](docs/rfdetr-pipeline.md) is the design record: what
the model measures, the sweeps behind each threshold, and why ~5,000 lines of
identity inference could be deleted rather than fixed. The **How it works** page
(`/architecture`) is the same story in plain language, and
[`docs/architecture-decision.md`](docs/architecture-decision.md) covers
scalability (the 80-camera streaming path, TimescaleDB tiering).

## Layout

```
apps/frontend/       Vite + TanStack Router SPA, shadcn UI, oRPC client
apps/api-service/    Bun + Hono + oRPC, BullMQ pipeline, Drizzle + postgres.js
packages/api-contracts/  Shared, type-only oRPC router types
services/ml-service/     Python FastAPI, RF-DETR (uv-managed, GPU-ready)
data/                    Uploaded videos, thumbnails
docker-compose.yml       TimescaleDB (5433) and Redis (6379)
```

## Prerequisites

- Docker (TimescaleDB + Redis)
- Bun 1.2+
- Python 3.12 and [uv](https://docs.astral.sh/uv/) for the ML service
- ffmpeg on PATH

## Running

```bash
# 1. Infrastructure (TimescaleDB on :5433, Redis on :6379)
docker compose up -d

# 2. ML service on :8000. RFDETR_WEIGHTS must point at the fine-tuned
#    checkpoint (see services/ml-service/.env) — there is no fallback detector.
cd services/ml-service && uv run uvicorn app.main:app --port 8000

# 3. JS dependencies and dev servers (from the repo root)
bun install
bun run db:migrate          # first run only, on a fresh database
bun run dev                 # api-service on :8787, frontend on :3001
```

Open http://localhost:3001. The BullMQ queue dashboard is at
http://localhost:8787/admin/queues.

## Ports

| Service              | Port       |
| -------------------- | ---------- |
| Frontend (Vite)      | 3001       |
| API (Hono)           | 8787       |
| ML service (FastAPI) | 8000       |
| TimescaleDB          | 5433       |
| Redis                | 6379       |
| MinIO (S3 + console) | 9000, 9001 |

## Commands

| Command               | What it does                                    |
| --------------------- | ----------------------------------------------- |
| `bun run dev`         | Run api-service and frontend together via Turbo |
| `bun run build`       | Production build of every workspace             |
| `bun run typecheck`   | Typecheck every workspace                       |
| `bun run lint`        | oxlint across the repo                          |
| `bun run format`      | Format with oxfmt                               |
| `bun run db:migrate`  | Apply Drizzle migrations                        |
| `docker compose down` | Stop infrastructure (data persists in volumes)  |

## Configuration

The api-service reads `API_SERVICE__*` variables with sensible local defaults
(database on `localhost:5433`, Redis on `localhost:6379`, ML service on
`localhost:8000`). The frontend reads `FRONTEND__API_URL`, defaulting to
`http://localhost:8787`.

Video + thumbnail bytes are stored per `API_SERVICE__STORAGE_BACKEND`: `local`
(the default; writes into `DATA_DIR`) or `s3` (MinIO / S3 / R2 via Bun's native
S3 client, configured with `API_SERVICE__S3_ENDPOINT` / `_BUCKET` / `_ACCESS_KEY`
/ `_SECRET_KEY`). On-prem MinIO keeps student video on the school's own
infrastructure; the worker caches a local copy for ffmpeg and the ML service.
