import { closeDb } from "@api/lib/db";
import { env } from "@api/lib/env";
import { logger } from "@api/lib/logger";
import { closeQueues } from "@api/lib/queue";
import { closeRedis } from "@api/lib/redis";
import { startWorkers, stopWorkers } from "@api/jobs/workers";
import { createApp } from "@api/server/app";

startWorkers();

const app = createApp();
const server = Bun.serve({
  hostname: env.API_SERVICE__HOST,
  port: env.API_SERVICE__PORT,
  idleTimeout: 120,
  // Bun's default request-body cap is far below a real classroom-video upload,
  // so raise it to the same ceiling the /videos route enforces (default 4 GB).
  // Without this, Bun returns an empty-bodied 413 before the route ever runs.
  maxRequestBodySize: env.API_SERVICE__MAX_UPLOAD_BYTES,
  fetch: app.fetch,
});

logger.info({ url: `http://${env.API_SERVICE__HOST}:${server.port}` }, "api-service listening");

let shuttingDown = false;
async function shutdown(signal: string): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  logger.info({ signal }, "draining");
  server.stop();
  await stopWorkers();
  await closeQueues();
  await closeRedis();
  await closeDb();
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
