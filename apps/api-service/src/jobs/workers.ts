import { QueueEvents, Worker } from "bullmq";
import { JOB_NAMES, QUEUE_NAMES } from "@api/lib/constants";
import { logger } from "@api/lib/logger";
import { createBullConnection } from "@api/lib/redis";
import { processAnalyzeJob } from "@api/jobs/analyze-video";
import type { AnalyzeJobData } from "@api/lib/queue";

let worker: Worker<AnalyzeJobData> | undefined;
let queueEvents: QueueEvents | undefined;

export function startWorkers(): void {
  // Concurrency 1: the ML service is a single-worker queue, and one classroom
  // video already saturates the local GPU.
  worker = new Worker<AnalyzeJobData>(
    QUEUE_NAMES.VIDEO_ANALYSIS,
    async (job) => {
      if (job.name === JOB_NAMES.ANALYZE) return processAnalyzeJob(job);
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

  queueEvents = new QueueEvents(QUEUE_NAMES.VIDEO_ANALYSIS, { connection: createBullConnection() });
  logger.info("workers started");
}

export async function stopWorkers(): Promise<void> {
  await worker?.close();
  await queueEvents?.close();
}
