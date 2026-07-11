import { env } from "@api/lib/env";

const BASE = env.API_SERVICE__ML_SERVICE_URL;

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

export interface DataQuality {
  detections: number;
  frames: number;
  identities: number;
  raw_tracks: number;
  fragmentation: number;
  coverage: number;
  occupied_buckets: number;
  span_buckets: number;
  concurrent_peak: number;
  concurrent_typical: number;
  confidence: {
    overall: QualityTier;
    occupancy: QualityTier;
    identity: QualityTier;
    coverage: QualityTier;
    teacher: QualityTier;
  };
  notes: string[];
}

export interface AnalysisResultAnalytics {
  teacher_present_ms: number;
  teacher_board_ms: number | null;
  entries: number;
  exits: number;
  presence_intervals: [number, number][];
  board_intervals: [number, number][];
  entry_exit: { kind: string; ts_ms: number }[];
  occupancy: { ts_ms: number; students: number; teacher: boolean }[];
  avg_students: number | null;
  max_students: number | null;
  heatmap: { grid_w: number; grid_h: number; teacher: number[]; students: number[] };
  teacher_pointing_ms: number | null;
  teacher_writing_ms: number | null;
  teacher_board_near_ms: number | null;
  board_interactions: { kind: "pointing" | "writing" | "near"; start_ms: number; end_ms: number }[];
  data_quality?: DataQuality | null;
}

export interface AnalysisResult {
  video: AnalysisResultVideo;
  tracks: AnalysisResultTrack[];
  events: AnalysisResultEvent[];
  analytics: AnalysisResultAnalytics;
}

export interface MlJobStatus {
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  stage: "detecting" | "merging" | "deriving" | null;
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
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`ML ${path} failed: ${res.status} ${await readErrorBody(res)}`);
  return (await res.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
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
    sample_fps: input.sampleFps ?? 5,
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
