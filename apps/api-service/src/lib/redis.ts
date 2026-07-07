import type { ConnectionOptions } from "bullmq";
import IORedis from "ioredis";
import { env } from "@api/lib/env";

// BullMQ requires maxRetriesPerRequest:null; each primitive gets its own connection.
export function createBullConnection(): ConnectionOptions {
  return new IORedis(env.API_SERVICE__REDIS_URL, {
    maxRetriesPerRequest: null,
  }) as unknown as ConnectionOptions;
}

const redis = new IORedis(env.API_SERVICE__REDIS_URL, { maxRetriesPerRequest: null });

export async function pingRedis(): Promise<void> {
  const pong = await redis.ping();
  if (pong !== "PONG") throw new Error(`Unexpected Redis ping: ${pong}`);
}

export async function closeRedis(): Promise<void> {
  await redis.quit().catch(() => redis.disconnect());
}
