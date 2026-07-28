import { type JobsOptions, Queue } from "bullmq";
import { DEFAULT_JOB_OPTIONS, JOB_NAMES, QUEUE_NAMES, type QueueName } from "@api/lib/constants";
import { createBullConnection } from "@api/lib/redis";

export interface AnalyzeJobData {
  videoId: string;
  attemptId?: string;
  // "rederive" replays roles/events/analytics from stored detections without
  // re-running YOLO. It goes through the QUEUE rather than the HTTP handler
  // because a 37-minute lesson takes minutes to derive, and an HTTP request
  // cannot outlive Bun's idleTimeout -- the request died at 300s and surfaced
  // as "Re-derivation failed", indistinguishable from a real failure.
  mode?: "rederive";
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

export function enqueueRederive(videoId: string) {
  return videoAnalysisQueue.add(JOB_NAMES.ANALYZE, { videoId, mode: "rederive" });
}

export async function closeQueues(): Promise<void> {
  await Promise.allSettled(Object.values(queues).map((q) => q.close()));
}
