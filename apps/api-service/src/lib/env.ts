import { createEnv } from "@t3-oss/env-core";
import * as z from "zod";

export const env = createEnv({
  server: {
    NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
    API_SERVICE__HOST: z.string().default("0.0.0.0"),
    API_SERVICE__PORT: z.coerce.number().int().positive().default(8787),
    API_SERVICE__DATABASE_URL: z
      .url()
      .default("postgres://postgres:postgres@localhost:5433/classroom"),
    API_SERVICE__REDIS_URL: z.url().default("redis://localhost:6379"),
    // 127.0.0.1, not localhost: uvicorn binds IPv4-only, but Bun's fetch
    // resolves localhost to IPv6 (::1) first and does not fall back, so
    // "localhost" intermittently yields ConnectionRefused to the ML service.
    API_SERVICE__ML_SERVICE_URL: z.url().default("http://127.0.0.1:8000"),
    API_SERVICE__DATA_DIR: z
      .string()
      .default("/Users/vishnumallela/Desktop/stackai/classroom-surveillance/data"),
    API_SERVICE__MAX_UPLOAD_BYTES: z.coerce
      .number()
      .int()
      .positive()
      .default(4 * 1024 * 1024 * 1024),
    API_SERVICE__CORS_ORIGINS: z
      .string()
      .default("http://localhost:3001")
      .transform((v) =>
        v
          .split(",")
          .map((o) => o.trim())
          .filter(Boolean),
      ),
    API_SERVICE__QUEUE_DASHBOARD_USER: z.string().default("admin"),
    API_SERVICE__QUEUE_DASHBOARD_PASSWORD: z.string().default("admin"),
    // Durable storage for video + thumbnail bytes. "local" writes into DATA_DIR
    // (dev default). "s3" stores objects in MinIO / S3 / R2 (on-prem or cloud,
    // one S3 API): durable, shared across workers, keeps large media out of the
    // app's local disk. The worker still materializes a local copy because the
    // ML service + ffmpeg need a real file path.
    API_SERVICE__STORAGE_BACKEND: z.enum(["local", "s3"]).default("local"),
    API_SERVICE__S3_ENDPOINT: z.string().default("http://localhost:9000"),
    API_SERVICE__S3_BUCKET: z.string().default("luminary-videos"),
    API_SERVICE__S3_ACCESS_KEY: z.string().default("minioadmin"),
    API_SERVICE__S3_SECRET_KEY: z.string().default("minioadmin"),
    API_SERVICE__S3_REGION: z.string().default("us-east-1"),
  },
  runtimeEnv: process.env,
  emptyStringAsUndefined: true,
});
