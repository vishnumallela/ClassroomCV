import * as z from "zod";
import {
  getAppSettings,
  mlServiceUrl,
  POD_DEFAULTS,
  podSpec,
  setAppSetting,
} from "@api/lib/app-settings";
import { DEFAULT_SCHOOL_TIMEZONE, isValidTimezone } from "@api/lib/school-time";
import { env } from "@api/lib/env";
import { inspectImage } from "@api/lib/registry";
import * as runpod from "@api/lib/runpod";
import { base } from "@api/orpc/base";

/** Mask a secret for display: enough to recognize, never enough to use. */
function mask(value: string | undefined): string | null {
  if (!value) return null;
  return value.length <= 8 ? "••••" : `${value.slice(0, 4)}…${value.slice(-4)}`;
}

/**
 * A connection string with the password blanked. The host and database name
 * are the parts worth checking on the Settings page; the password is not, and
 * echoing it back to the browser to fill a text input would put it in every
 * response, cache and devtools trace for no gain.
 */
function maskDsn(value: string | undefined): string | null {
  if (!value) return null;
  return value.replace(/:\/\/([^:@/]+):[^@]*@/u, "://$1:••••@");
}

export const settingsRouter = {
  get: base.handler(async () => {
    const s = await getAppSettings();
    const spec = await podSpec();
    const { DATABASE_URL: _dsn, PUBLIC_KEY: _key, ...safeEnv } = spec.env;
    return {
      runpodApiKeyMasked: mask(s.runpodApiKey),
      runpodPodId: s.runpodPodId ?? null,
      mlServiceUrl: s.mlServiceUrl ?? null,
      mlServiceUrlEffective: await mlServiceUrl(),
      mlServiceUrlDefault: env.API_SERVICE__ML_SERVICE_URL,
      gpuAutoStart: s.gpuAutoStart === "true",
      gpuAutoStopMinutes: Number(s.gpuAutoStopMinutes ?? "0") || 0,
      gpuIdleAction: s.gpuIdleAction === "stop" ? ("stop" as const) : ("terminate" as const),

      // The pod spec, resolved: the UI shows what WOULD be created, not a set
      // of blank fields the user has to guess defaults for. The two secrets in
      // `env` are replaced with set/masked flags — this response reaches the
      // browser, and a pod spec is not a reason to ship a DB password there.
      spec: { ...spec, env: safeEnv },
      defaults: POD_DEFAULTS,
      sshPublicKeySet: Boolean(s.gpuSshPublicKey),
      mlDatabaseUrlMasked: maskDsn(s.mlDatabaseUrl),
      schoolTimezone: s.schoolTimezone ?? null,
      schoolTimezoneEffective: s.schoolTimezone ?? DEFAULT_SCHOOL_TIMEZONE,
    };
  }),

  update: base
    .input(
      z.object({
        // undefined = leave untouched; "" = clear (falls back to the default).
        runpodApiKey: z.string().trim().max(200).optional(),
        runpodPodId: z.string().trim().max(100).optional(),
        mlServiceUrl: z.union([z.literal(""), z.url()]).optional(),
        gpuAutoStart: z.boolean().optional(),
        gpuAutoStopMinutes: z
          .number()
          .int()
          .min(0)
          .max(24 * 60)
          .optional(),
        gpuIdleAction: z.enum(["terminate", "stop"]).optional(),

        gpuPodName: z.string().trim().max(100).optional(),
        gpuImage: z.string().trim().max(300).optional(),
        gpuTypeId: z.string().trim().max(120).optional(),
        gpuCount: z.number().int().min(1).max(8).optional(),
        gpuCloudType: z.enum(["SECURE", "COMMUNITY"]).optional(),
        gpuDataCenterId: z.string().trim().max(40).optional(),
        gpuNetworkVolumeId: z.string().trim().max(60).optional(),
        gpuVolumeMountPath: z.string().trim().max(200).optional(),
        gpuContainerDiskGb: z.number().int().min(10).max(2000).optional(),
        gpuMinVcpu: z.number().int().min(1).max(128).optional(),
        gpuCudaVersions: z.string().trim().max(100).optional(),
        gpuInterruptible: z.boolean().optional(),
        gpuSshPublicKey: z.string().trim().max(2000).optional(),

        mlWeightsPath: z.string().trim().max(400).optional(),
        mlBatch: z.number().int().min(1).max(128).optional(),
        mlResolution: z.number().int().min(320).max(1536).optional(),
        mlTensorrt: z.boolean().optional(),
        mlMediaAllowlist: z.string().trim().max(500).optional(),
        mlDatabaseUrl: z.string().trim().max(500).optional(),

        // IANA zone the timetable is kept in. Everything in Group A/B of
        // docs/teacher-measurements.md converts through it.
        schoolTimezone: z.string().trim().max(80).optional(),
      }),
    )
    .handler(async ({ input, errors }) => {
      // An OpenSSH public key is easy to paste wrong (the PRIVATE key, or a
      // truncated line). A bad one is invisible until the pod is up and refuses
      // the connection you needed to upload the checkpoint with.
      if (
        input.gpuSshPublicKey &&
        !/^(ssh-(rsa|ed25519|dss)|ecdsa-sha2-)\S*\s+\S+/u.test(input.gpuSshPublicKey)
      ) {
        throw errors.VALIDATION({
          message:
            'That does not look like an OpenSSH public key (expected "ssh-ed25519 AAAA… user@host"). ' +
            "Paste the contents of ~/.ssh/id_ed25519.pub — not the private key.",
        });
      }

      // A zone name Intl cannot resolve would make every punctuality figure
      // silently fall back to the default zone, which is worse than refusing it.
      if (input.schoolTimezone && !isValidTimezone(input.schoolTimezone)) {
        throw errors.VALIDATION({
          message: `"${input.schoolTimezone}" is not an IANA timezone (expected e.g. "Asia/Kolkata").`,
        });
      }

      const text = {
        runpodApiKey: input.runpodApiKey,
        runpodPodId: input.runpodPodId,
        mlServiceUrl: input.mlServiceUrl,
        gpuIdleAction: input.gpuIdleAction,
        gpuPodName: input.gpuPodName,
        gpuImage: input.gpuImage,
        gpuTypeId: input.gpuTypeId,
        gpuCloudType: input.gpuCloudType,
        gpuDataCenterId: input.gpuDataCenterId,
        gpuNetworkVolumeId: input.gpuNetworkVolumeId,
        gpuVolumeMountPath: input.gpuVolumeMountPath,
        gpuCudaVersions: input.gpuCudaVersions,
        gpuSshPublicKey: input.gpuSshPublicKey,
        mlWeightsPath: input.mlWeightsPath,
        mlMediaAllowlist: input.mlMediaAllowlist,
        mlDatabaseUrl: input.mlDatabaseUrl,
        schoolTimezone: input.schoolTimezone,
      } as const;
      for (const [key, value] of Object.entries(text)) {
        if (value !== undefined) {
          await setAppSetting(key as keyof typeof text, value || null);
        }
      }

      const numbers = {
        gpuAutoStopMinutes: input.gpuAutoStopMinutes,
        gpuCount: input.gpuCount,
        gpuContainerDiskGb: input.gpuContainerDiskGb,
        gpuMinVcpu: input.gpuMinVcpu,
        mlBatch: input.mlBatch,
        mlResolution: input.mlResolution,
      } as const;
      for (const [key, value] of Object.entries(numbers)) {
        if (value !== undefined) {
          await setAppSetting(key as keyof typeof numbers, value > 0 ? String(value) : null);
        }
      }

      const flags = {
        gpuAutoStart: input.gpuAutoStart,
        gpuInterruptible: input.gpuInterruptible,
        mlTensorrt: input.mlTensorrt,
      } as const;
      for (const [key, value] of Object.entries(flags)) {
        if (value !== undefined) {
          await setAppSetting(key as keyof typeof flags, value ? "true" : null);
        }
      }

      return { ok: true as const };
    }),
};

interface PodErrors {
  CONFLICT: (o: { message: string }) => Error;
  DEPENDENCY_UNAVAILABLE: (o: { message: string }) => Error;
}

/**
 * Map a RunPod failure onto an oRPC error. "Not configured" is a CONFLICT the
 * UI turns into a prompt, everything else is the provider being unavailable —
 * and RunPod's own message ("no instances available for the requested GPU") is
 * far more actionable than anything this layer could substitute, so it is
 * passed through verbatim.
 */
function toPodError(err: unknown, errors: PodErrors): Error {
  const message = err instanceof Error ? err.message : String(err);
  return runpod.isConfiguredError(err)
    ? errors.CONFLICT({ message })
    : errors.DEPENDENCY_UNAVAILABLE({ message });
}

export const gpuRouter = {
  /** GPU pod + ML service health in one call for the Settings page. */
  status: base.handler(async () => {
    let pod: runpod.PodStatus | null = null;
    let podError: string | null = null;
    let configured = true;
    try {
      pod = await runpod.getPodStatus();
    } catch (err) {
      if (runpod.isConfiguredError(err)) configured = false;
      else podError = err instanceof Error ? err.message : String(err);
    }

    let ml: { healthy: boolean; device?: string; model?: string; backend?: string } = {
      healthy: false,
    };
    try {
      const url = await mlServiceUrl();
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const body = (await res.json()) as { device?: string; model?: string; backend?: string };
        ml = { healthy: true, device: body.device, model: body.model, backend: body.backend };
      }
    } catch {
      // unreachable — reported as unhealthy
    }
    return { configured, pod, podError, ml };
  }),

  /**
   * Everything the provisioning form needs, in one call: what GPUs are
   * purchasable right now and at what price, which regions can hold a volume,
   * and which volumes already exist. Empty (not an error) without an API key,
   * so the page renders before anything is configured.
   */
  catalog: base.handler(async () => {
    try {
      const [gpus, dataCenters, volumes] = await Promise.all([
        runpod.listGpus(),
        runpod.listDataCenters(),
        runpod.listNetworkVolumes(),
      ]);
      return { configured: true, gpus, dataCenters, volumes, error: null as string | null };
    } catch (err) {
      if (runpod.isConfiguredError(err)) {
        return { configured: false, gpus: [], dataCenters: [], volumes: [], error: null };
      }
      return {
        configured: true,
        gpus: [],
        dataCenters: [],
        volumes: [],
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }),

  /**
   * What the configured image tag actually contains.
   *
   * Separate from `catalog` because it hits the image's registry, not RunPod,
   * and is the one check that catches a tag pointing at the wrong program —
   * which costs a rented GPU to discover any other way.
   */
  image: base.handler(async () => {
    const spec = await podSpec();
    return { image: spec.image, check: await inspectImage(spec.image) };
  }),

  /** Pods already on the account, so an existing one can be adopted. */
  pods: base.handler(async () => {
    try {
      return { pods: await runpod.listPods() };
    } catch {
      return { pods: [] as runpod.PodStatus[] };
    }
  }),

  createVolume: base
    .input(
      z.object({
        name: z.string().trim().min(1).max(60),
        sizeGb: z.number().int().min(10).max(4000),
        dataCenterId: z.string().trim().min(1).max(40),
      }),
    )
    .handler(async ({ input, errors }) => {
      try {
        const volume = await runpod.createNetworkVolume(
          input.name,
          input.sizeGb,
          input.dataCenterId,
        );
        // Select it immediately: a volume created and then not chosen is the
        // most likely way to end up paying for two.
        await setAppSetting("gpuNetworkVolumeId", volume.id);
        await setAppSetting("gpuDataCenterId", input.dataCenterId);
        return { volume };
      } catch (err) {
        throw toPodError(err, errors);
      }
    }),

  /** Provision the pod. createPod records its id, which is what makes the ML
   * service URL resolve. */
  create: base.handler(async ({ errors }) => {
    try {
      return { pod: await runpod.createPod() };
    } catch (err) {
      throw toPodError(err, errors);
    }
  }),

  /** Destroy the pod. The volume, and therefore the checkpoint, survives. */
  terminate: base.handler(async ({ errors }) => {
    try {
      await runpod.terminatePod();
      return { ok: true as const };
    } catch (err) {
      throw toPodError(err, errors);
    }
  }),

  start: base.handler(async ({ errors }) => {
    try {
      return { pod: await runpod.startPod() };
    } catch (err) {
      throw toPodError(err, errors);
    }
  }),

  stop: base.handler(async ({ errors }) => {
    try {
      return { pod: await runpod.stopPod() };
    } catch (err) {
      throw toPodError(err, errors);
    }
  }),
};
