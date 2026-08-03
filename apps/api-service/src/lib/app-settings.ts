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
] as const;
export type SettingKey = (typeof SETTING_KEYS)[number];

const CACHE_TTL_MS = 10_000;
let cache: { at: number; values: Partial<Record<SettingKey, string>> } | null = null;

export async function getAppSettings(): Promise<Partial<Record<SettingKey, string>>> {
  if (cache && Date.now() - cache.at < CACHE_TTL_MS) return cache.values;
  const rows = await db.select().from(appSettings);
  const values: Partial<Record<SettingKey, string>> = {};
  for (const row of rows) {
    if ((SETTING_KEYS as readonly string[]).includes(row.key)) {
      values[row.key as SettingKey] = row.value;
    }
  }
  cache = { at: Date.now(), values };
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
}

/** ML service base URL: the Settings-page value wins over the env default, so
 * pointing the app at a fresh RunPod pod never needs a redeploy. */
export async function mlServiceUrl(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  return (settings.mlServiceUrl || env.API_SERVICE__ML_SERVICE_URL).replace(/\/+$/, "");
}
