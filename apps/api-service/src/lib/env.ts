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
    API_SERVICE__ML_SERVICE_URL: z.url().default("http://localhost:8000"),
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
  },
  runtimeEnv: process.env,
  emptyStringAsUndefined: true,
});
