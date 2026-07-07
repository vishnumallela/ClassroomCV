import { type JobsOptions, Queue } from "bullmq";
import { DEFAULT_JOB_OPTIONS, JOB_NAMES, QUEUE_NAMES, type QueueName } from "@api/lib/constants";
import { createBullConnection } from "@api/lib/redis";

export interface AnalyzeJobData {
  videoId: string;
  attemptId?: string;
}

const videoAnalysisQueue = new Queue<AnalyzeJobData>(QUEUE_NAMES.VIDEO_ANALYSIS, {
  connection: createBullConnection(),
  defaultJobOptions: DEFAULT_JOB_OPTIONS as JobsOptions,
});

export const queues: Record<QueueName, Queue> = {
  [QUEUE_NAMES.VIDEO_ANALYSIS]: videoAnalysisQueue,
};

export function enqueueAnalysis(data: AnalyzeJobData) {
  return videoAnalysisQueue.add(JOB_NAMES.ANALYZE, data);
}

export async function closeQueues(): Promise<void> {
  await Promise.allSettled(Object.values(queues).map((q) => q.close()));
}
