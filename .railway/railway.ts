/**
 * Luminary on Railway — the whole project as code.
 *
 * Five resources: TimescaleDB, Redis, MinIO (S3-compatible object storage for
 * video bytes), the Bun/Hono api-service and the Vite SPA behind Caddy. The
 * ML pipeline is deliberately NOT here: it needs a GPU, so it runs on an
 * on-demand RunPod pod that the app's Settings page starts and stops
 * (docs/runpod-gpu-deployment.md). The api reaches it over the URL stored in
 * app settings, which is why nothing below pins an ML host.
 *
 *   railway config plan     preview changes (safe)
 *   railway config apply    reconcile the project with this file
 *
 * Application code deploys itself: `api` and `web` are connected to the
 * GitHub repo, so Railway builds every push to main that matches their watch
 * patterns. This file only describes infrastructure and wiring.
 */
import {
  defineRailway,
  github,
  image,
  preserve,
  project,
  redis,
  service,
  volume,
} from "railway/iac";

const REPO = "vishnumallela/ClassroomCV";
const BRANCH = "main";

/** Object key prefix the api-service writes video + thumbnail bytes under. */
const MEDIA_BUCKET = "luminary-videos";

/**
 * Secrets are platform-managed, never generated from here. A `generator:`
 * re-runs on EVERY `config apply`, which silently rotates a live credential —
 * it rotated the Postgres password after initdb had already baked the old one
 * in, locking the api out of its own database. preserve() keeps whatever
 * Railway holds.
 *
 * Set them once, out of band (values never enter this repo):
 *
 *   railway variables set POSTGRES_PASSWORD=… --service Timescale
 *   railway variables set MINIO_ROOT_PASSWORD=… --service minio
 *   railway variables set API_SERVICE__ADMIN_PASSWORD=… --service api
 *   railway variables set API_SERVICE__QUEUE_DASHBOARD_PASSWORD=… --service api
 */

const PG_USER = "luminary";
const PG_DB = "classroom";

export default defineRailway(() => {
  // Railway's managed Postgres has no TimescaleDB, and the schema needs it:
  // detection_events is a hypertable with compression + retention policies and
  // a continuous aggregate (apps/api-service/drizzle/0003_storage_tiering.sql).
  //
  // Declared as a plain image service rather than database("…", "postgres"),
  // which provisions Railway's own Postgres image first and only then swaps in
  // the custom one — initdb has already run by that point, so the cluster ends
  // up with the wrong major version and no timescaledb in
  // shared_preload_libraries.
  const dbData = volume("timescale-pgdata", { sizeMB: 50000, region: "sfo" });
  const db = service("Timescale", {
    source: image("timescale/timescaledb:latest-pg17"),
    volumeMounts: { "/var/lib/postgresql/data": dbData },
    // A TCP proxy, because the ML service on RunPod is off-platform and writes
    // detections straight to the database.
    tcp: [5432],
    env: {
      // Postgres refuses to initdb into a non-empty directory, and Railway's
      // volume mount point is not empty (lost+found), so the cluster lives one
      // level down.
      PGDATA: "/var/lib/postgresql/data/pgdata",
      POSTGRES_USER: PG_USER,
      POSTGRES_DB: PG_DB,
      POSTGRES_PASSWORD: preserve(),
      DATABASE_URL: `postgres://${PG_USER}:\${{POSTGRES_PASSWORD}}@\${{RAILWAY_PRIVATE_DOMAIN}}:5432/${PG_DB}`,
      DATABASE_PUBLIC_URL: `postgres://${PG_USER}:\${{POSTGRES_PASSWORD}}@\${{RAILWAY_TCP_PROXY_DOMAIN}}:\${{RAILWAY_TCP_PROXY_PORT}}/${PG_DB}`,
    },
  });

  // BullMQ's broker: the upload → analyse → derive pipeline.
  const cache = redis("Redis");

  // On-prem-style object storage. The api-service talks to it with Bun's S3
  // client, and it is what lets a *remote* GPU worker read a lesson: the
  // worker gets a presigned URL, never a path on the api's disk.
  const minioData = volume("minio-data", { sizeMB: 50000, region: "sfo" });
  const minio = service("minio", {
    source: image("minio/minio:latest"),
    // MinIO's default command only prints help, so the server command is
    // explicit. On the single-drive backend a top-level directory *is* a
    // bucket, so mkdir is the idempotent "create bucket if missing".
    // [::] and not :9000 — MinIO resolves the bare form to IPv4 only, which
    // Railway's proxy cannot reach (502) and sibling services cannot dial.
    start: `/bin/sh -c "mkdir -p /data/${MEDIA_BUCKET} && exec minio server /data --address [::]:9000"`,
    volumeMounts: { "/data": minioData },
    env: {
      // Only the S3 API is wanted here. Left on, MinIO also opens its console
      // on a *random* high port, which is a second listener for Railway's
      // proxy to pick from — and picking it answers 502 on every S3 call.
      MINIO_BROWSER: "off",
      PORT: "9000",
      MINIO_ROOT_USER: "luminary",
      MINIO_ROOT_PASSWORD: preserve(),
    },
  });

  // DATA_DIR. On the s3 backend this is a *cache*, not the source of truth:
  // ffprobe/ffmpeg need a real file path, so the worker materialises objects
  // here before probing them. It still has to survive restarts.
  const mediaCache = volume("api-media-cache", { sizeMB: 20000, region: "sfo" });

  const api = service("api", {
    source: github(REPO, { branch: BRANCH }),
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "apps/api-service/Dockerfile",
      watchPatterns: [
        "apps/api-service/**",
        "packages/**",
        "package.json",
        "bun.lock",
        ".dockerignore",
        // This file too: it owns the variables baked into or read by the
        // image, so a change here has to reach a new build.
        ".railway/railway.ts",
      ],
    },
    start: "bun run start",
    // Migrations run before the new deployment takes traffic, so a schema
    // change can never be served by the old image or half-applied by two
    // replicas racing on boot.
    preDeploy: "bun run db:migrate",
    healthcheck: "/health",
    healthcheckTimeout: 300,
    volumeMounts: { "/data": mediaCache },
    env: {
      NODE_ENV: "production",
      // Railway routes public traffic to $PORT; the app binds it explicitly.
      PORT: "8787",
      API_SERVICE__PORT: "8787",
      // "::" and not "0.0.0.0": Railway's private network is IPv6-only, so a
      // v4-only listener is unreachable from sibling services.
      API_SERVICE__HOST: "::",
      API_SERVICE__DATABASE_URL: db.env.DATABASE_URL,
      API_SERVICE__REDIS_URL: cache.env.REDIS_URL,
      API_SERVICE__DATA_DIR: "/data",
      API_SERVICE__STORAGE_BACKEND: "s3",
      // The PUBLIC MinIO origin on purpose, not minio.railway.internal: the
      // presigned URLs signed against this endpoint are handed to the RunPod
      // GPU worker, which is off-platform and cannot resolve a private domain.
      // The bucket itself stays credentialed; only time-limited links escape.
      API_SERVICE__S3_ENDPOINT: "https://${{minio.RAILWAY_PUBLIC_DOMAIN}}",
      API_SERVICE__S3_BUCKET: MEDIA_BUCKET,
      API_SERVICE__S3_ACCESS_KEY: "${{minio.MINIO_ROOT_USER}}",
      API_SERVICE__S3_SECRET_KEY: "${{minio.MINIO_ROOT_PASSWORD}}",
      API_SERVICE__S3_REGION: "us-east-1",
      // The browser never uses this: it reaches the API same-origin through
      // web's /api proxy, so no request carries an Origin header. It only
      // covers anything hitting the api's own domain directly. Written as a
      // raw reference so the two services stay acyclic in code.
      API_SERVICE__CORS_ORIGINS: "https://${{web.RAILWAY_PUBLIC_DOMAIN}}",
      // Gates every route. Never leave this empty on a public URL: uploads,
      // deletes, the RunPod key and the GPU start/stop buttons sit behind it.
      API_SERVICE__ADMIN_PASSWORD: preserve(),
      API_SERVICE__QUEUE_DASHBOARD_USER: "admin",
      API_SERVICE__QUEUE_DASHBOARD_PASSWORD: preserve(),
    },
  });

  const web = service("web", {
    source: github(REPO, { branch: BRANCH }),
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "apps/frontend/Dockerfile",
      watchPatterns: [
        "apps/frontend/**",
        "packages/**",
        "package.json",
        "bun.lock",
        ".dockerignore",
        ".railway/railway.ts",
      ],
    },
    env: {
      // Caddy binds this (apps/frontend/Caddyfile) and Railway's edge routes
      // the public domain to it.
      PORT: "8080",
      // This service's OWN domain, not the api's: Caddy proxies /api to the
      // api service (apps/frontend/Caddyfile) so the SPA and the API share an
      // origin and the session cookie is first-party. Pointing the SPA
      // straight at the api's domain makes every browser drop that cookie.
      // Vite inlines it at BUILD time, so it is a build arg on this image.
      FRONTEND__API_URL: "https://${{RAILWAY_PUBLIC_DOMAIN}}/api",
    },
  });

  return project("classroomcv", {
    resources: [db, dbData, cache, minio, minioData, mediaCache, api, web],
  });
});
