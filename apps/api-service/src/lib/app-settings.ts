import { eq } from "drizzle-orm";
import { appSettings } from "@api/db/schema";
import { db } from "@api/lib/db";
import { env } from "@api/lib/env";

/**
 * App settings stored in Postgres and edited on the Settings page. A short
 * cache keeps hot paths (every ML call reads the service URL) off the DB.
 */

export const SETTING_KEYS = [
  "runpodApiKey",
  "runpodPodId",
  "mlServiceUrl",
  // "true" = start the pod automatically when work is queued and the ML
  // service is unreachable.
  "gpuAutoStart",
  // Minutes of idle (no active/waiting/delayed jobs) after which the pod is
  // stopped automatically. "" / "0" disables.
  "gpuAutoStopMinutes",
] as const;
export type SettingKey = (typeof SETTING_KEYS)[number];

const CACHE_TTL_MS = 10_000;
let cache: { at: number; values: Partial<Record<SettingKey, string>> } | null = null;
// Bumped on every write. A read that started BEFORE a write must not
// repopulate the cache with its (possibly stale) rows after the write's
// invalidation — the generation check makes that in-flight result uncacheable.
let generation = 0;

export async function getAppSettings(): Promise<Partial<Record<SettingKey, string>>> {
  if (cache && Date.now() - cache.at < CACHE_TTL_MS) return cache.values;
  const startedAt = generation;
  const rows = await db.select().from(appSettings);
  const values: Partial<Record<SettingKey, string>> = {};
  for (const row of rows) {
    if ((SETTING_KEYS as readonly string[]).includes(row.key)) {
      values[row.key as SettingKey] = row.value;
    }
  }
  if (startedAt === generation) cache = { at: Date.now(), values };
  return values;
}

export async function setAppSetting(key: SettingKey, value: string | null): Promise<void> {
  if (value === null || value === "") {
    await db.delete(appSettings).where(eq(appSettings.key, key));
  } else {
    await db
      .insert(appSettings)
      .values({ key, value, updatedAt: new Date() })
      .onConflictDoUpdate({
        target: appSettings.key,
        set: { value, updatedAt: new Date() },
      });
  }
  cache = null;
  generation++;
}

/** The port the ml-service listens on inside the pod (services/ml-service/Dockerfile). */
const ML_POD_PORT = 8000;

/**
 * ML service base URL, resolved per call so re-pointing the app never needs a
 * redeploy. Three sources, in order:
 *
 * 1. An explicit Settings-page override.
 * 2. RunPod's HTTP proxy hostname, derived from the pod id. This is the one
 *    RunPod address that SURVIVES stop/start — the public IP and the direct
 *    TCP port mappings are reassigned every time the pod starts, so anything
 *    built from those goes stale on the first stop. Deriving it means the pod
 *    id is the only thing anyone has to configure, and the autopilot's
 *    stop/start cycle needs no follow-up edit.
 * 3. The deployment default (local dev).
 *
 * A pod *migration* is the exception: RunPod moves the pod to another host and
 * issues a new id, at which point the id in Settings is what changes.
 */
export async function mlServiceUrl(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  const explicit = settings.mlServiceUrl?.trim();
  if (explicit) return explicit.replace(/\/+$/, "");

  const podId = settings.runpodPodId?.trim();
  if (podId) return `https://${podId}-${ML_POD_PORT}.proxy.runpod.net`;

  return env.API_SERVICE__ML_SERVICE_URL.replace(/\/+$/, "");
}
