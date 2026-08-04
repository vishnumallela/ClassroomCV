import { getAppSettings } from "@api/lib/app-settings";
import { logger } from "@api/lib/logger";

/**
 * RunPod REST client for ONE on-demand GPU pod (never serverless: the ML
 * service is a long-lived FastAPI process with a TensorRT engine cached on
 * the pod's volume, and an on-demand pod is what the Settings page can stop
 * so the meter stops with it).
 *
 * Pod stop/start recreates the container from its image on the next start,
 * which is also how CI ships a new ml-service build (push :latest, then
 * stop/start via this same API).
 */

const BASE = "https://rest.runpod.io/v1";

export interface PodStatus {
  id: string;
  name: string | null;
  /** RunPod desiredStatus: RUNNING | EXITED | TERMINATED (and transitions). */
  desiredStatus: string;
  costPerHr: number | null;
  gpuTypeId: string | null;
  publicIp: string | null;
  portMappings: Record<string, number> | null;
}

class RunpodNotConfigured extends Error {
  constructor() {
    super("RunPod is not configured. Set the API key and pod id in Settings.");
  }
}

async function credentials(): Promise<{ key: string; podId: string }> {
  const settings = await getAppSettings();
  const key = settings.runpodApiKey ?? "";
  const podId = settings.runpodPodId ?? "";
  if (!key || !podId) throw new RunpodNotConfigured();
  return { key, podId };
}

async function call<T>(
  key: string,
  method: "GET" | "POST",
  path: string,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`RunPod ${method} ${path} failed: ${res.status} ${body.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

function toStatus(pod: Record<string, unknown>): PodStatus {
  return {
    id: String(pod.id ?? ""),
    name: (pod.name as string) ?? null,
    desiredStatus: String(pod.desiredStatus ?? "UNKNOWN"),
    costPerHr: typeof pod.costPerHr === "number" ? pod.costPerHr : null,
    gpuTypeId: (pod.gpuTypeId as string) ?? null,
    publicIp: (pod.publicIp as string) ?? null,
    portMappings: (pod.portMappings as Record<string, number>) ?? null,
  };
}

export function isConfiguredError(err: unknown): boolean {
  return err instanceof RunpodNotConfigured;
}

export async function getPodStatus(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  return toStatus(await call<Record<string, unknown>>(key, "GET", `/pods/${podId}`));
}

export async function startPod(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  logger.info({ podId }, "starting RunPod GPU pod");
  return toStatus(await call<Record<string, unknown>>(key, "POST", `/pods/${podId}/start`));
}

/** Stop (not terminate): the volume — model weight, TensorRT engine cache —
 * survives, and billing drops to volume storage only. */
export async function stopPod(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  logger.info({ podId }, "stopping RunPod GPU pod");
  return toStatus(await call<Record<string, unknown>>(key, "POST", `/pods/${podId}/stop`));
}
