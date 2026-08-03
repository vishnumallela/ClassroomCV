import * as z from "zod";
import { getAppSettings, mlServiceUrl, setAppSetting } from "@api/lib/app-settings";
import { env } from "@api/lib/env";
import * as runpod from "@api/lib/runpod";
import { base } from "@api/orpc/base";

/** Mask a secret for display: enough to recognize, never enough to use. */
function mask(value: string | undefined): string | null {
  if (!value) return null;
  return value.length <= 8 ? "••••" : `${value.slice(0, 4)}…${value.slice(-4)}`;
}

export const settingsRouter = {
  get: base.handler(async () => {
    const s = await getAppSettings();
    return {
      runpodApiKeyMasked: mask(s.runpodApiKey),
      runpodPodId: s.runpodPodId ?? null,
      mlServiceUrl: s.mlServiceUrl ?? null,
      mlServiceUrlEffective: await mlServiceUrl(),
      mlServiceUrlDefault: env.API_SERVICE__ML_SERVICE_URL,
    };
  }),

  update: base
    .input(
      z.object({
        // undefined = leave untouched; "" = clear.
        runpodApiKey: z.string().trim().max(200).optional(),
        runpodPodId: z.string().trim().max(100).optional(),
        mlServiceUrl: z
          .union([z.literal(""), z.url()])
          .optional(),
      }),
    )
    .handler(async ({ input }) => {
      if (input.runpodApiKey !== undefined) {
        await setAppSetting("runpodApiKey", input.runpodApiKey || null);
      }
      if (input.runpodPodId !== undefined) {
        await setAppSetting("runpodPodId", input.runpodPodId || null);
      }
      if (input.mlServiceUrl !== undefined) {
        await setAppSetting("mlServiceUrl", input.mlServiceUrl || null);
      }
      return { ok: true as const };
    }),
};

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

    let ml: { healthy: boolean; device?: string; model?: string } = { healthy: false };
    try {
      const url = await mlServiceUrl();
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const body = (await res.json()) as { device?: string; model?: string };
        ml = { healthy: true, device: body.device, model: body.model };
      }
    } catch {
      // unreachable — reported as unhealthy
    }
    return { configured, pod, podError, ml };
  }),

  start: base.handler(async ({ errors }) => {
    try {
      return { pod: await runpod.startPod() };
    } catch (err) {
      if (runpod.isConfiguredError(err)) {
        throw errors.CONFLICT({ message: "RunPod is not configured in Settings." });
      }
      throw errors.DEPENDENCY_UNAVAILABLE({
        message: err instanceof Error ? err.message : "RunPod start failed.",
      });
    }
  }),

  stop: base.handler(async ({ errors }) => {
    try {
      return { pod: await runpod.stopPod() };
    } catch (err) {
      if (runpod.isConfiguredError(err)) {
        throw errors.CONFLICT({ message: "RunPod is not configured in Settings." });
      }
      throw errors.DEPENDENCY_UNAVAILABLE({
        message: err instanceof Error ? err.message : "RunPod stop failed.",
      });
    }
  }),
};
