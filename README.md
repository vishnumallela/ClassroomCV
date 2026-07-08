# Luminary

_Every lesson, brought to light._

Upload a classroom recording and get trustworthy teaching analytics: teacher
presence, entries and exits, time at the board, circulation among the desks,
how the class settled, and student occupancy over time — each shown with an
honest confidence level, never a falsely precise number. No facial recognition,
no named students, aggregate insights only.

The detection pipeline (YOLO26-pose, BoT-SORT, CLIP re-identification, SAM 2
zone detection) runs in a durable background worker; the dashboard reads the
results over a typed API. Every analysed lesson also carries a **data-quality
report** (`services/ml-service/app/quality.py`) — coverage, tracker
fragmentation, and a re-identification-independent concurrent head count — so
the dashboard can say how much each figure can be trusted.

The **How it works** page (`/architecture`) explains the whole pipeline in plain
language, and [`docs/architecture-decision.md`](docs/architecture-decision.md)
is the SOTA + scalability decision record (detection/tracking upgrades, the
80-camera streaming path, and the TimescaleDB tiering for high-volume ingest).

## Layout

```
apps/frontend/       Vite + TanStack Router SPA, shadcn UI, oRPC client
apps/api-service/    Bun + Hono + oRPC, BullMQ pipeline, Drizzle + postgres.js
packages/api-contracts/  Shared, type-only oRPC router types
services/ml-service/     Python FastAPI, YOLO26-pose + SAM 2 (uv-managed, GPU-ready)
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

# 2. ML service on :8000
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
