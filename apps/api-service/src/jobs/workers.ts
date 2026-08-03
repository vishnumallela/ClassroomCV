import { DelayedError, Worker } from "bullmq";
import { JOB_NAMES, QUEUE_NAMES } from "@api/lib/constants";
import { mlServiceUrl } from "@api/lib/app-settings";
import { logger } from "@api/lib/logger";
import { createBullConnection } from "@api/lib/redis";
import { processAnalyzeJob } from "@api/jobs/analyze-video";
import type { AnalyzeJobData } from "@api/lib/queue";

let worker: Worker<AnalyzeJobData> | undefined;

// How long a job sleeps when the ML service is unreachable (the RunPod GPU is
// stopped or still booting). Delaying does NOT consume a retry attempt, so a
// lesson uploaded with the GPU off waits indefinitely and processes when the
// pod comes back — instead of burning all 5 attempts inside a minute.
const GPU_WAIT_DELAY_MS = 60_000;

async function mlReachable(): Promise<boolean> {
  try {
    const url = await mlServiceUrl();
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
    return res.ok;
  } catch {
    return false;
  }
}

export function startWorkers(): void {
  // Concurrency 1: the ML service is a single-worker queue, and one classroom
  // video already saturates the local GPU.
  worker = new Worker<AnalyzeJobData>(
    QUEUE_NAMES.VIDEO_ANALYSIS,
    async (job, token) => {
      if (job.name === JOB_NAMES.ANALYZE) {
        if (!(await mlReachable())) {
          logger.info(
            { jobId: job.id, videoId: job.data.videoId },
            "ML service unreachable (GPU off?); delaying job without burning an attempt",
          );
          await job.moveToDelayed(Date.now() + GPU_WAIT_DELAY_MS, token);
          throw new DelayedError();
        }
        return processAnalyzeJob(job);
      }
      throw new Error(`No processor registered for job "${job.name}"`);
    },
    { connection: createBullConnection(), concurrency: 1 },
  );

  worker.on("failed", (job, err) =>
    logger.error({ jobId: job?.id, attemptsMade: job?.attemptsMade, err }, "analysis job failed"),
  );
  worker.on("completed", (job) =>
    logger.info({ jobId: job.id, videoId: job.data.videoId }, "analysis job completed"),
  );
  worker.on("error", (err) => logger.error({ err }, "worker error"));

  logger.info("workers started");
}

export async function stopWorkers(): Promise<void> {
  await worker?.close();
}
