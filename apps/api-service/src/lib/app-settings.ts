import { eq } from "drizzle-orm";
import { appSettings } from "@api/db/schema";
import { db } from "@api/lib/db";
import { env } from "@api/lib/env";

/**
 * App settings stored in Postgres and edited on the Settings page. A short
 * cache keeps hot paths (every ML call reads the service URL) off the DB.
 *
 * The `gpu*` and `ml*` keys together are the whole RunPod pod specification —
 * everything that used to be typed into the RunPod web console. They live here
 * rather than in `env` so a pod can be re-provisioned on a different GPU, in a
 * different region, from a different image, without redeploying this service.
 */

export const SETTING_KEYS = [
  "runpodApiKey",
  /** Set by provisioning; cleared on terminate. Also accepts an adopted pod. */
  "runpodPodId",
  "mlServiceUrl",
  // "true" = provision/start the pod automatically when work is queued and the
  // ML service is unreachable.
  "gpuAutoStart",
  // Minutes of idle (no active/waiting/delayed jobs) after which the pod is
  // released automatically. "" / "0" disables.
  "gpuAutoStopMinutes",
  // What "release" means: "terminate" (default) destroys the pod, "stop" keeps
  // it. See gpuIdleAction below and runpod.stopPod's warning.
  "gpuIdleAction",

  // ── pod spec ──────────────────────────────────────────────────────────────
  "gpuPodName",
  "gpuImage",
  "gpuTypeId",
  "gpuCount",
  "gpuCloudType",
  "gpuDataCenterId",
  "gpuNetworkVolumeId",
  "gpuVolumeMountPath",
  "gpuContainerDiskGb",
  "gpuMinVcpu",
  "gpuCudaVersions",
  "gpuInterruptible",
  "gpuSshPublicKey",

  // ── ml-service environment handed to the pod at create time ───────────────
  "mlWeightsPath",
  "mlBatch",
  "mlResolution",
  "mlTensorrt",
  "mlMediaAllowlist",
  "mlDatabaseUrl",
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
 * Defaults for a pod that has never been configured. Chosen so that "add an API
 * key, pick a volume, press Create" produces a working GPU worker.
 *
 * The GPU default is an L4: video analytics here is decode-bound, not
 * GPU-bound (a 37-minute lesson peaked at 4.5 GB of 32 GB), so a bigger card
 * buys nothing. CUDA is pinned to 13.0/12.8 because uv.lock's torch 2.12.1 is a
 * cu13 build needing driver >= r580 — an unpinned pod can land on a 12.4 host
 * and silently run on CPU.
 *
 * THE IMAGE TAG IS NOT `:latest`, and that is deliberate. deploy-ml-runpod.yml
 * lets only `main` write `:latest`, and the RF-DETR rewrite is still on
 * `feat/rfdetr-pipeline`; `:latest` therefore still resolves to the previous
 * YOLO/identity-stack service — built 2026-08-06, declaring `MODEL_NAME` and
 * `IMGSZ` instead of `RFDETR_*`, and carrying no entrypoint and so no sshd. A
 * pod on it starts, reports RUNNING, ignores every RF-DETR variable it is
 * given, and cannot be shelled into to find out why.
 *
 * **On merging this branch to main, change this back to `:latest`** — at that
 * point `:latest` becomes the RF-DETR image and a branch tag would be the stale
 * one. Until then the Settings page inspects whatever tag is configured and
 * says which service it actually contains, so a wrong tag is caught before a
 * GPU is rented rather than after (see lib/registry.ts).
 */
export const POD_DEFAULTS = {
  name: "classroomcv-ml",
  image: "ghcr.io/vishnumallela/classroomcv/ml-service:feat-rfdetr-pipeline",
  gpuTypeId: "NVIDIA L4",
  gpuCount: 1,
  cloudType: "SECURE" as const,
  dataCenterId: "EU-RO-1",
  volumeMountPath: "/workspace",
  containerDiskInGb: 50,
  // vCPUs demanded PER GPU. RunPod's own default is 2 and this field was
  // missing from the spec entirely, so an L4 pod came up with a 5.1-core cgroup
  // — and the pipeline is decode-bound, not GPU-bound. Measured on a 37-minute
  // 1440p lesson: uvicorn pegged at 580% CPU while the GPU idled at 6%, because
  // H.264 is inter-frame coded so all 55,893 frames must be decoded to use the
  // 11,179 that get sampled. More cores is the one lever that moves this
  // without touching the decode path.
  //
  // 16 is a request, not a reservation: it narrows the eligible machines, so
  // drop it if a GPU shows as available yet nothing provisions.
  minVcpuPerGpu: 16,
  // Used only when NO network volume is attached: the pod gets its own disk at
  // volumeMountPath instead. Survives stop/start, dies with the pod.
  podVolumeGb: 50,
  // 13.0 ONLY. uv.lock pins torch 2.12.1, which resolves to a cu13 wheel and
  // needs driver >= r580; a CUDA 12.8 host (driver 570.x) cannot run it, so
  // torch.cuda.is_available() comes back False, the device resolves to cpu and
  // REQUIRE_DEVICE aborts the job — after the pod has already pulled the image,
  // downloaded the video and loaded the checkpoint. Listing 12.8 here only buys
  // the right to rent a machine that cannot work: measured on a pod that landed
  // on driver 570.195.03 and failed 7 minutes in.
  //
  // This narrows the eligible hosts. If nothing provisions, lower vCPU per GPU
  // or change GPU type — do NOT re-add 12.8.
  cudaVersions: ["13.0"],
  weightsPath: "/workspace/weights/rfdetr-medium.pth",
  batch: 16,
  resolution: 576,
} as const;

export interface PodSpec {
  name: string;
  image: string;
  gpuTypeId: string;
  gpuCount: number;
  cloudType: "SECURE" | "COMMUNITY";
  dataCenterId: string;
  networkVolumeId: string;
  volumeMountPath: string;
  containerDiskInGb: number;
  minVcpuPerGpu: number;
  /** Size of the pod's own volume, used only when networkVolumeId is empty. */
  podVolumeGb: number;
  allowedCudaVersions: string[];
  interruptible: boolean;
  env: Record<string, string>;
}

/**
 * The pod specification as configured, resolved to concrete values.
 *
 * `env` is what the container sees. DEVICE/REQUIRE_DEVICE are set explicitly
 * even though the image already defaults them: the pair is the only thing
 * standing between a mis-scheduled pod and a silent CPU run that bills ~20x the
 * wall-clock, and it should be visible in the pod's env in the RunPod console
 * when someone goes looking.
 */
export async function podSpec(): Promise<PodSpec> {
  const s = await getAppSettings();
  const mount = s.gpuVolumeMountPath?.trim() || POD_DEFAULTS.volumeMountPath;

  const podEnv: Record<string, string> = {
    DEVICE: "cuda",
    REQUIRE_DEVICE: "cuda",
    RFDETR_WEIGHTS: s.mlWeightsPath?.trim() || POD_DEFAULTS.weightsPath,
    RFDETR_BATCH: String(Number(s.mlBatch) || POD_DEFAULTS.batch),
    RFDETR_RESOLUTION: String(Number(s.mlResolution) || POD_DEFAULTS.resolution),
    RFDETR_TENSORRT: s.mlTensorrt === "true" ? "true" : "false",
    DATA_DIR: `${mount}/data`,
  };
  // Both are optional at create time — a pod can come up healthy and be pointed
  // at storage later — but an /analyze with neither set fails on the pod, so
  // they are surfaced as warnings by the preflight check rather than silently
  // omitted here.
  const allowlist = s.mlMediaAllowlist?.trim();
  if (allowlist) podEnv.MEDIA_URL_ALLOWLIST = allowlist;
  const databaseUrl = s.mlDatabaseUrl?.trim();
  if (databaseUrl) podEnv.DATABASE_URL = databaseUrl;
  // RunPod's own convention: templates read PUBLIC_KEY, and so does
  // docker-entrypoint.sh, which is how the checkpoint gets onto a fresh volume.
  const sshKey = s.gpuSshPublicKey?.trim();
  if (sshKey) podEnv.PUBLIC_KEY = sshKey;

  return {
    name: s.gpuPodName?.trim() || POD_DEFAULTS.name,
    image: s.gpuImage?.trim() || POD_DEFAULTS.image,
    gpuTypeId: s.gpuTypeId?.trim() || POD_DEFAULTS.gpuTypeId,
    gpuCount: Number(s.gpuCount) || POD_DEFAULTS.gpuCount,
    cloudType: s.gpuCloudType === "COMMUNITY" ? "COMMUNITY" : "SECURE",
    dataCenterId: s.gpuDataCenterId?.trim() || POD_DEFAULTS.dataCenterId,
    networkVolumeId: s.gpuNetworkVolumeId?.trim() ?? "",
    volumeMountPath: mount,
    containerDiskInGb: Number(s.gpuContainerDiskGb) || POD_DEFAULTS.containerDiskInGb,
    podVolumeGb: POD_DEFAULTS.podVolumeGb,
    minVcpuPerGpu: Number(s.gpuMinVcpu) || POD_DEFAULTS.minVcpuPerGpu,
    allowedCudaVersions: (s.gpuCudaVersions?.trim() || POD_DEFAULTS.cudaVersions.join(","))
      .split(",")
      .map((v) => v.trim())
      .filter(Boolean),
    interruptible: s.gpuInterruptible === "true",
    env: podEnv,
  };
}

/** "terminate" destroys the pod when idle; "stop" keeps it (and its host pin). */
export async function gpuIdleAction(): Promise<"terminate" | "stop"> {
  const s = await getAppSettings();
  return s.gpuIdleAction === "stop" ? "stop" : "terminate";
}

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
 *    release/provision cycle needs no follow-up edit: the id is rewritten in
 *    Settings the moment a new pod is created.
 * 3. The deployment default (local dev).
 */
export async function mlServiceUrl(): Promise<string> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  const explicit = settings.mlServiceUrl?.trim();
  if (explicit) return explicit.replace(/\/+$/, "");

  const podId = settings.runpodPodId?.trim();
  if (podId) return `https://${podId}-${ML_POD_PORT}.proxy.runpod.net`;

  return env.API_SERVICE__ML_SERVICE_URL.replace(/\/+$/, "");
}
