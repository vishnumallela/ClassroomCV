import { getAppSettings, podSpec, setAppSetting } from "@api/lib/app-settings";
import { logger } from "@api/lib/logger";

/**
 * RunPod client: catalog, network volumes, and the lifecycle of the ONE
 * on-demand GPU pod that runs ml-service.
 *
 * Never serverless — the ML service is a long-lived FastAPI process whose
 * fine-tuned checkpoint (and any TensorRT engine) lives on a network volume,
 * and an on-demand pod is what the Settings page can provision and destroy so
 * the meter stops with it.
 *
 * TWO BASE URLS, because RunPod split its API and neither half is complete:
 *
 *   rest.runpod.io/v1   pods CRUD + start/stop/restart, /networkvolumes,
 *                       /billing/pods. NO catalog — /v1/gputypes 400s.
 *   api.runpod.io/v2    /catalog/gpus and /catalog/datacenters, with live
 *                       per-hour prices and stock. NO start/stop — those 404.
 *
 * So provisioning reads v2 to find out what is purchasable and writes to v1 to
 * buy it. Both were probed against a live key before this was written; do not
 * "tidy" them into one host.
 */

const REST = "https://rest.runpod.io/v1";
const CATALOG = "https://api.runpod.io/v2";

/** Container port ml-service listens on (services/ml-service/Dockerfile). */
export const ML_POD_PORT = 8000;

export interface PodStatus {
  id: string;
  name: string | null;
  /** RunPod desiredStatus: RUNNING | EXITED | TERMINATED (and transitions). */
  desiredStatus: string;
  costPerHr: number | null;
  gpuTypeId: string | null;
  gpuCount: number | null;
  image: string | null;
  dataCenterId: string | null;
  networkVolumeId: string | null;
  publicIp: string | null;
  portMappings: Record<string, number> | null;
  /** UTC ISO. Null before the pod has ever started. */
  lastStartedAt: string | null;
}

class RunpodNotConfigured extends Error {
  constructor(message = "RunPod is not configured. Add the API key in Settings.") {
    super(message);
  }
}

/** A RunPod HTTP failure, carrying the status so 404 can be told from 500. */
class RunpodHttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export function isConfiguredError(err: unknown): boolean {
  return err instanceof RunpodNotConfigured;
}

/** The API key alone — enough to browse the catalog and create a pod. */
async function apiKey(): Promise<string> {
  const key = (await getAppSettings()).runpodApiKey ?? "";
  if (!key) throw new RunpodNotConfigured();
  return key;
}

/** Key + the id of an existing pod — required to inspect or destroy one. */
async function credentials(): Promise<{ key: string; podId: string }> {
  const settings = await getAppSettings();
  const key = settings.runpodApiKey ?? "";
  const podId = settings.runpodPodId ?? "";
  if (!key) throw new RunpodNotConfigured();
  if (!podId) throw new RunpodNotConfigured("No GPU pod yet. Create one in Settings.");
  return { key, podId };
}

async function call<T>(
  base: string,
  key: string,
  method: "GET" | "POST" | "DELETE",
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${base}${path}`, {
    method,
    headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (res.status === 401 || res.status === 403) {
    throw new Error("RunPod rejected the API key — it needs read/write scope.");
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new RunpodHttpError(
      res.status,
      `RunPod ${method} ${path} failed: ${res.status} ${text.slice(0, 300)}`,
    );
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ── catalog ─────────────────────────────────────────────────────────────────

export interface GpuType {
  id: string;
  name: string;
  memory: number;
  manufacturer: string;
  price: { secure: number; community: number };
  maxCount: { secure: number; community: number };
}

export interface DataCenter {
  id: string;
  name: string;
  region: string;
  networkVolumeTypes: string[];
}

/**
 * Purchasable GPUs, cheapest first.
 *
 * The raw catalog carries an `unknown` placeholder (0 GB, $0) that passes an
 * availability check but cannot be provisioned, plus cards with no stock on
 * either tier. Offering either produces a create call that fails after the user
 * has already committed to the choice, so they are filtered here.
 */
export async function listGpus(): Promise<GpuType[]> {
  const key = await apiKey();
  const body = await call<{ gpus?: GpuType[] }>(CATALOG, key, "GET", "/catalog/gpus");
  return (body.gpus ?? [])
    .filter((g) => g.id !== "unknown" && g.memory > 0)
    .filter((g) => g.maxCount.secure > 0 || g.maxCount.community > 0)
    .toSorted(
      (a, b) => (a.price.community || a.price.secure) - (b.price.community || b.price.secure),
    );
}

/**
 * Datacenters that can hold a network volume.
 *
 * A volume is pinned to its region permanently and a pod can only mount one in
 * its own region, so a region without volume support is never a valid choice
 * for this app — the checkpoint has to live somewhere the pod can reach.
 */
export async function listDataCenters(): Promise<DataCenter[]> {
  const key = await apiKey();
  const body = await call<{ dataCenters?: DataCenter[] }>(
    CATALOG,
    key,
    "GET",
    "/catalog/datacenters",
  );
  return (body.dataCenters ?? []).filter((d) => d.networkVolumeTypes?.length);
}

// ── network volumes ─────────────────────────────────────────────────────────

export interface NetworkVolume {
  id: string;
  name: string;
  size: number;
  dataCenterId: string;
}

export async function listNetworkVolumes(): Promise<NetworkVolume[]> {
  const key = await apiKey();
  // v1 returns a bare array with `dataCenterId`; v2's /network-volumes wraps it
  // and calls the same field `dataCenter`. Use v1 so create and list agree.
  return call<NetworkVolume[]>(REST, key, "GET", "/networkvolumes");
}

export async function createNetworkVolume(
  name: string,
  sizeGb: number,
  dataCenterId: string,
): Promise<NetworkVolume> {
  const key = await apiKey();
  logger.info({ name, sizeGb, dataCenterId }, "creating RunPod network volume");
  return call<NetworkVolume>(REST, key, "POST", "/networkvolumes", {
    name,
    size: sizeGb,
    dataCenterId,
  });
}

// ── pods ────────────────────────────────────────────────────────────────────

/**
 * Map a v1 pod onto PodStatus. Every field below was read off a real running
 * pod, because the shape is not guessable: the GPU MODEL lives on `machine`
 * while the GPU COUNT is top-level, the image is `imageName` (not `image`), and
 * there is no `gpu` object at all. Reading `pod.gpuTypeId`, as this once did,
 * always yielded null and left the Settings page showing a blank card.
 */
function toStatus(pod: Record<string, unknown>): PodStatus {
  const machine = (pod.machine ?? {}) as Record<string, unknown>;
  const volume = (pod.networkVolume ?? {}) as Record<string, unknown>;
  return {
    id: String(pod.id ?? ""),
    name: (pod.name as string) ?? null,
    desiredStatus: String(pod.desiredStatus ?? "UNKNOWN"),
    costPerHr: typeof pod.costPerHr === "number" ? pod.costPerHr : null,
    gpuTypeId: (machine.gpuTypeId as string) ?? null,
    gpuCount: typeof pod.gpuCount === "number" ? pod.gpuCount : null,
    image: ((pod.imageName ?? pod.image) as string) ?? null,
    dataCenterId: (machine.dataCenterId as string) ?? null,
    networkVolumeId: ((pod.networkVolumeId ?? volume.id) as string) ?? null,
    publicIp: (pod.publicIp as string) ?? null,
    portMappings: (pod.portMappings as Record<string, number>) ?? null,
    lastStartedAt: (pod.lastStartedAt as string) ?? null,
  };
}

/**
 * Sub-objects RunPod OMITS unless asked for. `includeMachine` and
 * `includeNetworkVolume` both default to false, and without them the pod comes
 * back with no `machine` and no `networkVolume` at all — so the GPU model,
 * region and attached volume read as null however healthy the pod is, and the
 * status card renders blank. Verified against a real running pod.
 */
const POD_INCLUDES = "?includeMachine=true&includeNetworkVolume=true";

/**
 * The configured pod.
 *
 * A 404 is not an outage — it means the pod was destroyed elsewhere (the RunPod
 * console, a savings-plan expiry, someone else's terminate). The saved id is
 * then worse than useless: mlServiceUrl() would keep deriving a proxy hostname
 * with nothing behind it, so every job waits forever on a dead address. Forget
 * the id and report it as "no pod", which is the state that autopilot knows how
 * to fix by provisioning a replacement.
 */
export async function getPodStatus(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  try {
    return toStatus(
      await call<Record<string, unknown>>(REST, key, "GET", `/pods/${podId}${POD_INCLUDES}`),
    );
  } catch (err) {
    if (err instanceof RunpodHttpError && err.status === 404) {
      logger.warn({ podId }, "saved RunPod pod no longer exists — clearing the id");
      await setAppSetting("runpodPodId", null);
      throw new RunpodNotConfigured("The saved GPU pod no longer exists. Create one in Settings.");
    }
    throw err;
  }
}

/** Every pod on the account — used to adopt one that this app did not create. */
export async function listPods(): Promise<PodStatus[]> {
  const key = await apiKey();
  const pods = await call<Record<string, unknown>[]>(REST, key, "GET", `/pods${POD_INCLUDES}`);
  return (pods ?? []).map((pod) => toStatus(pod));
}

export interface CreatePodOptions {
  /** Overrides the saved spec, so the UI can create without saving first. */
  name?: string;
}

/**
 * Provision the ml-service pod from the spec held in Settings.
 *
 * Everything that used to be typed into the RunPod console lives in
 * `podSpec()`: image, GPU, cloud tier, datacenter, volume, disk and the
 * ml-service environment. Two fields in the body are load-bearing and were
 * each learned the expensive way:
 *
 *  - `allowedCudaVersions`. uv.lock pins torch 2.12.1, which resolves to a cu13
 *    wheel and needs driver >= r580. Without this pin, RunPod is free to place
 *    the pod on a driver-12.4 host; the container then starts, reports healthy,
 *    and runs on `device: cpu` at roughly 20x the wall-clock. REQUIRE_DEVICE
 *    catches it at load, but only after you have paid for the boot.
 *  - `ports`. 8000/http is what RunPod's proxy hostname front-ends, and that
 *    hostname is the address the API service talks to. 22/tcp is how the
 *    checkpoint gets onto a fresh volume — the image ships sshd for exactly
 *    that reason (see docker-entrypoint.sh).
 */
export async function createPod(options: CreatePodOptions = {}): Promise<PodStatus> {
  const key = await apiKey();
  const spec = await podSpec();
  const name = options.name ?? spec.name;

  // This app manages exactly ONE pod, so adopt an orphan rather than renting a
  // second GPU beside it. The window is real: if RunPod creates the pod but the
  // response is lost, the id never reaches Settings, and the next autopilot
  // tick would provision again — silently doubling the bill with nothing to
  // show which pod is which.
  const orphan = (await listPods().catch(() => [])).find(
    (p) => p.name === name && p.desiredStatus !== "TERMINATED",
  );
  if (orphan) {
    logger.warn({ podId: orphan.id, name }, "a pod with this name already exists — adopting it");
    await setAppSetting("runpodPodId", orphan.id);
    return orphan;
  }

  const body: Record<string, unknown> = {
    name,
    imageName: spec.image,
    computeType: "GPU",
    gpuTypeIds: [spec.gpuTypeId],
    gpuCount: spec.gpuCount,
    gpuTypePriority: "availability",
    cloudType: spec.cloudType,
    volumeMountPath: spec.volumeMountPath,
    containerDiskInGb: spec.containerDiskInGb,
    // The pipeline is decode-bound: without this the pod gets RunPod's default
    // allocation and the container cgroup lands around 5 cores, which pegs at
    // 100% while the GPU sits at 6%.
    minVCPUPerGPU: spec.minVcpuPerGpu,
    allowedCudaVersions: spec.allowedCudaVersions,
    ports: [`${ML_POD_PORT}/http`, "22/tcp"],
    interruptible: spec.interruptible,
    env: spec.env,
  };

  // A network volume is the better home for the checkpoint — it outlives the
  // pod, so terminate-on-idle costs nothing to undo. But it is not required,
  // and demanding one was wrong: RunPod refuses to create a network volume
  // below a $5 account balance, which would leave someone unable to run at all
  // on an account that can perfectly well rent a GPU.
  //
  // Without one, `volumeInGb` gives the pod its own disk at the same mount
  // path. That survives stop/start but NOT terminate, so the checkpoint has to
  // be re-uploaded for each new pod. The Settings page says so rather than
  // letting it be discovered as a failed /analyze.
  //
  // THE REGION PIN RIDES ON THE VOLUME, and only on it. A volume is welded to
  // its datacenter and a pod must mount it from inside the same one, so with a
  // volume the region is not a preference — it is a constraint. With no volume
  // there is nothing to be near, and pinning anyway is how you get "no L4
  // available" while the catalog cheerfully reports nine: `maxCount` is a
  // GLOBAL count, so a card can be plentiful worldwide and absent from the one
  // datacenter the create was nailed to. Let RunPod place it instead.
  if (spec.networkVolumeId) {
    body.networkVolumeId = spec.networkVolumeId;
    body.dataCenterIds = [spec.dataCenterId];
    body.dataCenterPriority = "custom";
  } else {
    body.volumeInGb = spec.podVolumeGb;
    body.dataCenterPriority = "availability";
  }

  logger.info(
    {
      gpu: spec.gpuTypeId,
      dc: spec.networkVolumeId ? spec.dataCenterId : "any (no volume to be near)",
      image: spec.image,
      cloud: spec.cloudType,
    },
    "creating RunPod GPU pod",
  );
  const pod = toStatus(await call<Record<string, unknown>>(REST, key, "POST", "/pods", body));
  // Recording the id here rather than at each call site is deliberate: the id
  // is what mlServiceUrl() derives the proxy hostname from, so a pod created
  // without it recorded is a pod the rest of the app cannot reach.
  await setAppSetting("runpodPodId", pod.id);
  return pod;
}

/**
 * Destroy the pod. The network volume — checkpoint, TensorRT engine cache,
 * video scratch — is a separate resource and survives, so the replacement pod
 * comes up with its weights already in place.
 */
export async function terminatePod(): Promise<void> {
  const { key, podId } = await credentials();
  logger.info({ podId }, "terminating RunPod GPU pod");
  await call<void>(REST, key, "DELETE", `/pods/${podId}`);
  // Forget the id in the same breath. Left set, mlServiceUrl() keeps resolving
  // to a proxy hostname with nothing behind it, so every job would wait on a
  // dead address instead of falling back to the configured default.
  await setAppSetting("runpodPodId", null);
}

export async function startPod(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  logger.info({ podId }, "starting RunPod GPU pod");
  return toStatus(await call<Record<string, unknown>>(REST, key, "POST", `/pods/${podId}/start`));
}

/**
 * Stop without destroying.
 *
 * Cheaper in principle — billing drops to volume storage — but READ THIS FIRST:
 * a stopped pod stays pinned to its host machine, and that machine's GPU is
 * re-rented to someone else in the meantime. The restart then fails with "not
 * enough free GPUs on the host machine" and the pod is stranded; this has
 * already cost this project one pod. Terminate-and-recreate is the reliable
 * cycle, which is why it is what autopilot does by default.
 */
export async function stopPod(): Promise<PodStatus> {
  const { key, podId } = await credentials();
  logger.info({ podId }, "stopping RunPod GPU pod (may fail to restart — see stopPod docs)");
  return toStatus(await call<Record<string, unknown>>(REST, key, "POST", `/pods/${podId}/stop`));
}
