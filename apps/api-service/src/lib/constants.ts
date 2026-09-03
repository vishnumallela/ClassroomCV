export const QUEUE_NAMES = {
  VIDEO_ANALYSIS: "video-analysis",
  // Its own queue, not a second job on the video one. Audio needs no GPU, so
  // it must not inherit that queue's concurrency of 1 (one lesson saturates the
  // card) nor its "wait for the pod" delay. Transcription is someone else's
  // compute: ten lessons can be in flight while the GPU chews through one.
  AUDIO_ANALYSIS: "audio-analysis",
} as const;
export type QueueName = (typeof QUEUE_NAMES)[keyof typeof QUEUE_NAMES];

export const JOB_NAMES = {
  ANALYZE: "analyze-video",
  ANALYZE_AUDIO: "analyze-audio",
} as const;
export type JobName = (typeof JOB_NAMES)[keyof typeof JOB_NAMES];

export const DEFAULT_JOB_OPTIONS = {
  attempts: 5,
  backoff: { type: "exponential", delay: 2_000 },
  removeOnComplete: { age: 3_600, count: 500 },
  removeOnFail: { age: 86_400, count: 1_000 },
} as const;
