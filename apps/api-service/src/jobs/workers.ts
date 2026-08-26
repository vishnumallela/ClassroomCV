import { DelayedError, Worker } from "bullmq";
import { JOB_NAMES, QUEUE_NAMES } from "@api/lib/constants";
import { getAppSettings, gpuIdleAction, mlServiceUrl } from "@api/lib/app-settings";
import { updateStatus } from "@api/db/queries";
import { logger } from "@api/lib/logger";
import {
  createPod,
  getPodStatus,
  isConfiguredError,
  type PodStatus,
  startPod,
  stopPod,
  terminatePod,
} from "@api/lib/runpod";
import { createBullConnection } from "@api/lib/redis";
import { processAnalyzeJob } from "@api/jobs/analyze-video";
import { type AnalyzeJobData, queues } from "@api/lib/queue";

let worker: Worker<AnalyzeJobData> | undefined;
let autopilotTimer: ReturnType<typeof setInterval> | undefined;

// How long a job sleeps when the ML service is unreachable (the RunPod GPU is
// stopped or still booting). Delaying does NOT consume a retry attempt, so a
// lesson uploaded with the GPU off waits indefinitely and processes when the
// pod comes back — instead of burning all 5 attempts inside a minute.
const GPU_WAIT_DELAY_MS = 60_000;
// Auto-start is attempted at most this often: a booting pod (image pull,
// weights load, fp16 trace) shows unreachable for several delay cycles, and
// hammering the create endpoint each cycle would rent a second GPU.
const AUTO_START_MIN_INTERVAL_MS = 3 * 60_000;
const AUTOPILOT_TICK_MS = 60_000;

let lastAutoStartAt = 0;
// Last moment the queue was non-idle (job finished, or work observed). The
// auto-stop clock measures from here, so the pod survives short gaps between
// lessons of a batch.
let lastActivityAt = Date.now();

async function mlReachable(): Promise<boolean> {
  try {
    const url = await mlServiceUrl();
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Queued work with no GPU serving it: bring one up (autopilot).
 *
 * Three cases, and the first is what makes this app self-sufficient: with no
 * pod at all — because the last idle window terminated it, or because nobody
 * has ever created one — this PROVISIONS one from the spec in Settings. That
 * is the point of holding the spec there rather than in the RunPod console:
 * recovery does not require a human with a browser.
 */
async function maybeAutoStart(): Promise<void> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  if (settings.gpuAutoStart !== "true") return;
  if (Date.now() - lastAutoStartAt < AUTO_START_MIN_INTERVAL_MS) return;
  lastAutoStartAt = Date.now();
  try {
    let pod: PodStatus;
    try {
      pod = await getPodStatus();
    } catch (err) {
      // No pod: either none was ever configured, or getPodStatus just cleared
      // a stale id because the pod had been destroyed. Provision a replacement.
      // Without an API key there is nothing to provision from, so that stays a
      // no-op rather than a create attempt that cannot succeed.
      if (!isConfiguredError(err) || !settings.runpodApiKey) throw err;
      logger.info("autopilot: queued work and no GPU pod — provisioning one");
      await createPod();
      return;
    }
    if (pod.desiredStatus !== "RUNNING") {
      logger.info({ podId: pod.id }, "autopilot: queued work with GPU stopped — starting pod");
      await startPod();
    }
  } catch (err) {
    logger.warn({ err }, "autopilot: bring-up failed (will retry)");
  }
}

/** Idle for longer than the configured window: release the GPU (autopilot). */
async function autoStopTick(): Promise<void> {
  try {
    const settings = await getAppSettings();
    const idleMinutes = Number(settings.gpuAutoStopMinutes ?? "0") || 0;
    if (idleMinutes <= 0) return;
    const queue = queues[QUEUE_NAMES.VIDEO_ANALYSIS];
    const counts = await queue.getJobCounts("active", "waiting", "delayed", "prioritized");
    const pending =
      (counts.active ?? 0) +
      (counts.waiting ?? 0) +
      (counts.delayed ?? 0) +
      (counts.prioritized ?? 0);
    if (pending > 0) {
      lastActivityAt = Date.now();
      return;
    }
    if (Date.now() - lastActivityAt < idleMinutes * 60_000) return;
    const pod = await getPodStatus();
    if (pod.desiredStatus !== "RUNNING") return;

    // TERMINATE, not stop, by default. A stopped pod stays pinned to its host
    // machine while that machine's GPU is re-rented to someone else, and the
    // restart then fails with "not enough free GPUs on the host machine" — a
    // pod stranded that way has to be destroyed and recreated anyway, except
    // now it happens with a lesson already waiting. Terminating costs the same
    // (billing ends either way) and the checkpoint is on the network volume,
    // which outlives the pod. "stop" stays selectable for anyone who wants the
    // container layer preserved and accepts that risk.
    const action = await gpuIdleAction();
    logger.info(
      { podId: pod.id, idleMinutes, action },
      "autopilot: queue idle past the window — releasing GPU",
    );
    if (action === "stop") await stopPod();
    else await terminatePod();
  } catch {
    // Not configured / RunPod unreachable: nothing to release.
  }
}

export function startWorkers(): void {
  // Concurrency 1: the ML service is a single-worker queue, and one classroom
  // video already saturates the local GPU.
  worker = new Worker<AnalyzeJobData>(
    QUEUE_NAMES.VIDEO_ANALYSIS,
    async (job, token) => {
      if (job.name === JOB_NAMES.ANALYZE) {
        lastActivityAt = Date.now();
        if (!(await mlReachable())) {
          logger.info(
            { jobId: job.id, videoId: job.data.videoId },
            "ML service unreachable (GPU off?); delaying job without burning an attempt",
          );
          // Let the dashboard say WHY nothing is happening. Benign if a newer
          // run owns the video: it only runs when ML is reachable, and its
          // first step immediately stamps its own status.
          await updateStatus(job.data.videoId, { status: "waiting_gpu" }).catch(() => undefined);
          await maybeAutoStart();
          await job.moveToDelayed(Date.now() + GPU_WAIT_DELAY_MS, token);
          throw new DelayedError();
        }
        return processAnalyzeJob(job);
      }
      throw new Error(`No processor registered for job "${job.name}"`);
    },
    { connection: createBullConnection(), concurrency: 1 },
  );

  worker.on("failed", (job, err) => {
    lastActivityAt = Date.now();
    logger.error({ jobId: job?.id, attemptsMade: job?.attemptsMade, err }, "analysis job failed");
  });
  worker.on("completed", (job) => {
    lastActivityAt = Date.now();
    logger.info({ jobId: job.id, videoId: job.data.videoId }, "analysis job completed");
  });
  worker.on("error", (err) => logger.error({ err }, "worker error"));

  autopilotTimer = setInterval(() => void autoStopTick(), AUTOPILOT_TICK_MS);

  logger.info("workers started");
}

export async function stopWorkers(): Promise<void> {
  if (autopilotTimer) clearInterval(autopilotTimer);
  await worker?.close();
}
