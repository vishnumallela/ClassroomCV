export const QUEUE_NAMES = { VIDEO_ANALYSIS: "video-analysis" } as const;
export type QueueName = (typeof QUEUE_NAMES)[keyof typeof QUEUE_NAMES];

export const JOB_NAMES = { ANALYZE: "analyze-video" } as const;
export type JobName = (typeof JOB_NAMES)[keyof typeof JOB_NAMES];

export const DEFAULT_JOB_OPTIONS = {
  attempts: 5,
  backoff: { type: "exponential", delay: 2_000 },
  removeOnComplete: { age: 3_600, count: 500 },
  removeOnFail: { age: 86_400, count: 1_000 },
} as const;
