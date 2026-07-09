import { type Job, UnrecoverableError } from "bullmq";
import { dirname, join } from "node:path";
import {
  countDetectionEvents,
  getVideo,
  getZones,
  hasZoneKind,
  insertZone,
  replaceDerived,
  updateStatus,
  updateVideo,
  type VideoRow,
} from "@api/db/queries";
import { mkdir } from "node:fs/promises";
import { generateThumbnail, probeVideo } from "@api/lib/media";
import { logger } from "@api/lib/logger";
import { mlDetectBoard, mlGetJob, mlGetJobResult, mlStartAnalysis } from "@api/lib/ml";
import { isS3, presignGet, putLocalFile } from "@api/lib/storage";

// The bytes source for ffprobe/ffmpeg/the ML worker. On s3 this is a presigned
// URL (valid 6 h, long enough for a slow analysis) so nothing downloads the
// whole video onto the API node: ffprobe reads only the header, ffmpeg only a
// seeked frame, and the ML worker fetches its own local copy. On local it is
// just the file path.
function mediaSource(filePath: string): string {
  return isS3 ? (presignGet(filePath, 6 * 60 * 60) ?? filePath) : filePath;
}
import type { AnalyzeJobData } from "@api/lib/queue";

const POLL_INTERVAL_MS = 5_000;
// Only RUNNING polls burn the 2h processing budget; queue waits are capped at 24h.
const MAX_RUNNING_POLLS = (2 * 60 * 60) / 5;
const MAX_TOTAL_POLLS = (24 * 60 * 60) / 5;

function runOwnsVideo(video: VideoRow, attemptId: string | undefined, jobId: string): boolean {
  const stored = video.workflowRunId;
  if (stored === null) return true;
  if (attemptId !== undefined && stored === attemptId) return true;
  return stored === jobId;
}

async function requireCurrentRun(
  videoId: string,
  attemptId: string | undefined,
  jobId: string,
  step: string,
): Promise<VideoRow> {
  const video = await getVideo(videoId);
  if (!video) throw new UnrecoverableError(`video ${videoId} was deleted before ${step}`);
  if (!runOwnsVideo(video, attemptId, jobId)) {
    throw new UnrecoverableError(`run superseded before ${step}`);
  }
  return video;
}

function clampProgress(value: number): number {
  return Math.min(0.99, Math.max(0, value));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function probeStep(
  videoId: string,
  attemptId: string | undefined,
  jobId: string,
): Promise<void> {
  const video = await requireCurrentRun(videoId, attemptId, jobId, "probe");
  await updateStatus(videoId, { status: "probing", progress: 0.02 });
  const source = mediaSource(video.filePath);
  const meta = await probeVideo(source);

  let thumbnailPath: string | undefined;
  try {
    const mark = meta.durationMs ? (meta.durationMs / 1000) * 0.1 : 1;
    const out = join(dirname(video.filePath), "thumb.jpg");
    await mkdir(dirname(out), { recursive: true });
    if (await generateThumbnail(source, out, mark)) {
      thumbnailPath = out;
      await putLocalFile(out).catch((err) =>
        logger.warn({ err, videoId }, "thumbnail upload to object store failed (non-fatal)"),
      );
    }
  } catch (err) {
    logger.warn({ err, videoId }, "thumbnail generation failed (non-fatal)");
  }

  await updateVideo(videoId, {
    durationMs: meta.durationMs,
    fps: meta.fps,
    width: meta.width,
    height: meta.height,
    ...(thumbnailPath ? { thumbnailPath } : {}),
    status: "analyzing",
    progress: 0.05,
  });
}

async function detectBoardStep(
  videoId: string,
  attemptId: string | undefined,
  jobId: string,
): Promise<void> {
  await requireCurrentRun(videoId, attemptId, jobId, "detect-board");
  if (await hasZoneKind(videoId, "board")) return;
  try {
    const video = await getVideo(videoId);
    if (!video) return;
    const res = await mlDetectBoard(videoId, mediaSource(video.filePath));
    if (res.polygon && res.confidence >= 0.5) {
      await requireCurrentRun(videoId, attemptId, jobId, "detect-board insert");
      await insertZone(videoId, {
        kind: "board",
        polygon: res.polygon,
        meta: { auto: true, confidence: res.confidence, method: res.method },
      });
    }
  } catch (err) {
    if (err instanceof UnrecoverableError) throw err;
    logger.warn({ err, videoId }, "board auto-detect failed (continuing without board)");
  }
}

async function startAnalysisStep(
  videoId: string,
  attemptId: string | undefined,
  jobId: string,
): Promise<string> {
  const video = await requireCurrentRun(videoId, attemptId, jobId, "start-analysis");
  const zones = await getZones(videoId);
  const runTokens = [attemptId, jobId].filter((t): t is string => Boolean(t));
  return mlStartAnalysis({
    videoId,
    videoPath: mediaSource(video.filePath),
    sampleFps: 5,
    zones,
    idempotencyKey: `${videoId}:${attemptId ?? "initial"}`,
    runTokens,
  });
}

async function pollUntilDone(
  videoId: string,
  mlJobId: string,
  attemptId: string | undefined,
  jobId: string,
  job: Job<AnalyzeJobData>,
): Promise<void> {
  let runningPolls = 0;
  for (let attempt = 0; attempt < MAX_TOTAL_POLLS; attempt++) {
    await requireCurrentRun(videoId, attemptId, jobId, "poll");
    const status = await mlGetJob(mlJobId);
    if (status.status === "done") return;
    if (status.status === "failed") {
      throw new UnrecoverableError(`ML analysis failed: ${status.error ?? "unknown error"}`);
    }
    const progress = clampProgress(status.progress);
    await updateStatus(videoId, {
      status: status.stage === "deriving" ? "deriving" : "analyzing",
      progress,
    });
    await job.updateProgress(progress);
    if (status.status === "running") {
      runningPolls++;
      if (runningPolls >= MAX_RUNNING_POLLS) {
        throw new UnrecoverableError("ML analysis did not complete within 2 hours of processing");
      }
    }
    await sleep(POLL_INTERVAL_MS);
  }
  throw new UnrecoverableError("ML analysis did not complete within 24 hours");
}

async function ingestStep(
  videoId: string,
  mlJobId: string,
  attemptId: string | undefined,
  jobId: string,
): Promise<void> {
  const result = await mlGetJobResult(mlJobId);
  const video = await requireCurrentRun(videoId, attemptId, jobId, "ingest");

  const probed = await countDetectionEvents(videoId);
  if (probed > 0 && result.tracks.length === 0) {
    throw new UnrecoverableError(`0 tracks while ${probed} detection rows exist`);
  }
  if (probed > 0 && (!result.video || result.video.duration_ms === 0)) {
    throw new UnrecoverableError(`duration_ms=0 while ${probed} detection rows exist`);
  }

  const meta = result.video;
  if (meta && meta.duration_ms > 0 && (video.durationMs === null || video.durationMs <= 0)) {
    await updateVideo(videoId, {
      durationMs: meta.duration_ms,
      fps: video.fps ?? (meta.fps > 0 ? meta.fps : null),
      width: video.width ?? (meta.width > 0 ? meta.width : null),
      height: video.height ?? (meta.height > 0 ? meta.height : null),
    });
  }

  await replaceDerived(videoId, result, { markDone: true });
}

export async function processAnalyzeJob(job: Job<AnalyzeJobData>): Promise<void> {
  const { videoId, attemptId } = job.data;
  const jobId = String(job.id);
  try {
    await probeStep(videoId, attemptId, jobId);
    await detectBoardStep(videoId, attemptId, jobId);
    const mlJobId = await startAnalysisStep(videoId, attemptId, jobId);
    await pollUntilDone(videoId, mlJobId, attemptId, jobId, job);
    await ingestStep(videoId, mlJobId, attemptId, jobId);
  } catch (err) {
    // Mark failed only when this is the terminal attempt, and only if we still
    // own the video, so a superseding reanalyze is never stamped 'failed'.
    const terminal =
      err instanceof UnrecoverableError || job.attemptsMade + 1 >= (job.opts.attempts ?? 1);
    if (terminal) {
      const video = await getVideo(videoId);
      if (video && runOwnsVideo(video, attemptId, jobId)) {
        await updateStatus(videoId, {
          status: "failed",
          progress: 0,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
    throw err;
  }
}
