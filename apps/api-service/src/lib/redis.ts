import type { ConnectionOptions } from "bullmq";
import IORedis from "ioredis";
import { env } from "@api/lib/env";

// ioredis defaults to family:4, which cannot resolve a host that only has an
// AAAA record — exactly the case on Railway, whose private network is
// IPv6-only. family:0 lets the resolver pick either, so the same code reaches
// redis.railway.internal in production and 127.0.0.1 in dev.
const CONNECTION = { maxRetriesPerRequest: null, family: 0 } as const;

// BullMQ requires maxRetriesPerRequest:null; each primitive gets its own connection.
export function createBullConnection(): ConnectionOptions {
  return new IORedis(env.API_SERVICE__REDIS_URL, {
    ...CONNECTION,
  }) as unknown as ConnectionOptions;
}

const redis = new IORedis(env.API_SERVICE__REDIS_URL, { ...CONNECTION });

export async function pingRedis(): Promise<void> {
  const pong = await redis.ping();
  if (pong !== "PONG") throw new Error(`Unexpected Redis ping: ${pong}`);
}

export async function closeRedis(): Promise<void> {
  await redis.quit().catch(() => redis.disconnect());
}
