import { DelayedError, Worker } from "bullmq";
import { JOB_NAMES, QUEUE_NAMES } from "@api/lib/constants";
import { getAppSettings, mlServiceUrl } from "@api/lib/app-settings";
import { updateStatus } from "@api/db/queries";
import { logger } from "@api/lib/logger";
import { getPodStatus, startPod, stopPod } from "@api/lib/runpod";
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
// Auto-start is attempted at most this often: a booting pod (weights load,
// TensorRT warmup) shows unreachable for several delay cycles, and hammering
// the start endpoint each cycle helps nothing.
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

/** Queued work with the GPU off: optionally start the pod (autopilot). */
async function maybeAutoStart(): Promise<void> {
  const settings = await getAppSettings().catch(() => ({}) as Record<string, never>);
  if (settings.gpuAutoStart !== "true") return;
  if (Date.now() - lastAutoStartAt < AUTO_START_MIN_INTERVAL_MS) return;
  lastAutoStartAt = Date.now();
  try {
    const pod = await getPodStatus();
    if (pod.desiredStatus !== "RUNNING") {
      logger.info({ podId: pod.id }, "autopilot: queued work with GPU off — starting pod");
      await startPod();
    }
  } catch (err) {
    logger.warn({ err }, "autopilot: auto-start failed (will retry)");
  }
}

/** Idle for longer than the configured window: stop the pod (autopilot). */
async function autoStopTick(): Promise<void> {
  try {
    const settings = await getAppSettings();
    const idleMinutes = Number(settings.gpuAutoStopMinutes ?? "0") || 0;
    if (idleMinutes <= 0) return;
    const queue = queues[QUEUE_NAMES.VIDEO_ANALYSIS];
    const counts = await queue.getJobCounts("active", "waiting", "delayed", "prioritized");
    const pending =
      (counts.active ?? 0) + (counts.waiting ?? 0) + (counts.delayed ?? 0) + (counts.prioritized ?? 0);
    if (pending > 0) {
      lastActivityAt = Date.now();
      return;
    }
    if (Date.now() - lastActivityAt < idleMinutes * 60_000) return;
    const pod = await getPodStatus();
    if (pod.desiredStatus === "RUNNING") {
      logger.info(
        { podId: pod.id, idleMinutes },
        "autopilot: queue idle past the window — stopping pod",
      );
      await stopPod();
    }
  } catch {
    // Not configured / RunPod unreachable: nothing to stop.
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
