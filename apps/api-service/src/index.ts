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
  // Bun.serve caps request bodies at 128 MB by default, which rejects real
  // classroom recordings with a 413 long before the handler's own size check.
  // Lift it to the app-level upload limit so handleUpload owns the decision.
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
