import { env } from "@api/lib/env";

import { mlServiceUrl } from "@api/lib/app-settings";

export interface MlZone {
  kind: string;
  polygon: [number, number][];
}

export interface AnalysisResultVideo {
  duration_ms: number;
  fps: number;
  width: number;
  height: number;
}

export interface AnalysisResultTrack {
  track_no: number;
  role: string;
  role_confidence: number | null;
  first_ms: number;
  last_ms: number;
  meta: Record<string, unknown> | null;
}

export interface AnalysisResultEvent {
  kind: string;
  video_ts_ms: number;
  track_no: number | null;
}

export type QualityTier = "high" | "medium" | "low";

// Mirrors services/ml-service/app/quality.py. The old identity/fragmentation
// signals described how hard an appearance merge had to work to reassemble one
// person out of tracker fragments; the detector now names the teacher, so what
// can go wrong is coverage, continuity and detection confidence.
export interface DataQuality {
  detections: number;
  frames: number;
  sampled_frames: number;
  coverage: number;
  mean_confidence: number;
  breaks: number;
  longest_gap_ms: number;
  confidence: {
    overall: QualityTier;
    coverage: QualityTier;
    continuity: QualityTier;
    teacher: QualityTier;
  };
  notes: string[];
}

export interface AnalysisResultAnalytics {
  teacher_present_ms: number;
  // null = the input was absent (no board zone; or a /rederive that replayed
  // teacher-only stored rows and so never saw the pointing/writing classes).
  // 0 = measured, and it did not happen. See AnalyticsOut in app/models.py.
  teacher_board_ms: number | null;
  teacher_pointing_ms?: number | null;
  teacher_writing_ms?: number | null;
  entries: number;
  exits: number;
  presence_intervals: [number, number][];
  board_intervals: [number, number][];
  pointing_intervals?: [number, number][];
  writing_intervals?: [number, number][];
  entry_exit: { kind: string; ts_ms: number }[];
  heatmap: { grid_w: number; grid_h: number; teacher: number[] };
  data_quality?: DataQuality | null;
}

/** A zone the analysis placed from its own detections, for us to persist. */
export interface ProposedZone {
  kind: "board" | "door";
  polygon: [number, number][];
  confidence: number;
  method: string;
  frame_ts_ms: number;
}

export interface AnalysisResult {
  video: AnalysisResultVideo;
  tracks: AnalysisResultTrack[];
  events: AnalysisResultEvent[];
  analytics: AnalysisResultAnalytics;
  // Optional: a pod on an older image does not send it.
  proposed_zones?: ProposedZone[];
}

export interface MlJobStatus {
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  // No 'merging' stage: there are no identity fragments to merge.
  stage: "detecting" | "deriving" | null;
  error: string | null;
}

export interface BoardDetectResult {
  polygon: [number, number][] | null;
  confidence: number;
  method: string;
  frame_ts_ms: number;
}

async function readErrorBody(res: Response): Promise<string> {
  try {
    return (await res.text()).slice(0, 500);
  } catch {
    return "";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  // Base URL resolves per call (Settings-page override wins over env), so
  // re-pointing at a fresh RunPod pod takes effect without a redeploy.
  const res = await fetch(`${await mlServiceUrl()}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`ML ${path} failed: ${res.status} ${await readErrorBody(res)}`);
  return (await res.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${await mlServiceUrl()}${path}`);
  if (!res.ok) throw new Error(`ML ${path} failed: ${res.status} ${await readErrorBody(res)}`);
  return (await res.json()) as T;
}

export interface StartAnalysisInput {
  videoId: string;
  videoPath: string;
  sampleFps?: number;
  zones: MlZone[];
  idempotencyKey?: string;
  runTokens?: string[];
}

export async function mlStartAnalysis(input: StartAnalysisInput): Promise<string> {
  // JSON.stringify drops the undefined optionals, which disables the ML-side
  // fence/idempotency for direct calls that pass neither.
  const res = await post<{ job_id?: string }>("/analyze", {
    video_id: input.videoId,
    video_path: input.videoPath,
    sample_fps: input.sampleFps ?? env.API_SERVICE__SAMPLE_FPS,
    zones: input.zones,
    idempotency_key: input.idempotencyKey,
    run_tokens: input.runTokens,
  });
  if (!res.job_id) throw new Error("ML /analyze returned no job_id");
  return res.job_id;
}

export function mlGetJob(jobId: string): Promise<MlJobStatus> {
  return get(`/jobs/${jobId}`);
}

export function mlGetJobResult(jobId: string): Promise<AnalysisResult> {
  return get(`/jobs/${jobId}/result`);
}

export function mlDetectBoard(videoId: string, videoPath: string): Promise<BoardDetectResult> {
  return post("/detect-board", { video_id: videoId, video_path: videoPath });
}

export function mlDetectDoor(videoId: string, videoPath: string): Promise<BoardDetectResult> {
  return post("/detect-door", { video_id: videoId, video_path: videoPath });
}

export function mlRederive(videoId: string, zones: MlZone[]): Promise<AnalysisResult> {
  return post("/rederive", { video_id: videoId, zones });
}
