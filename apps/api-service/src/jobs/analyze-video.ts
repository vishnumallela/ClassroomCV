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
import { localDateInSchoolTz, periodOffsets, schoolTimezone } from "@api/lib/school-time";
import { logger } from "@api/lib/logger";
import { mlGetJob, mlGetJobResult, mlHealth, mlStartAnalysis } from "@api/lib/ml";
import { isS3, presignGet, putLocalFile } from "@api/lib/storage";

// The bytes source for ffprobe/ffmpeg/the ML worker. On s3 this is a presigned
// URL (valid 6 h, long enough for a slow analysis) so nothing downloads the
// whole video onto the API node: ffprobe reads only the header, ffmpeg only a
// seeked frame, and the ML worker fetches its own local copy. On local it is
// just the file path. Exported: every ML call site (incl. the interactive
// zone-detect route) must use this, or a remote GPU pod receives a local path
// it cannot read.
export function mediaSource(filePath: string): string {
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

  // Seed the lesson's wall-clock anchor and date from the container, but never
  // over a value already on the row: a re-analysis must not silently discard a
  // correction someone typed after watching the recording. Both stay editable.
  const lessonSeed: { recordingStartedAt?: Date; lessonDate?: string } = {};
  if (meta.recordingStartedAt && video.recordingStartedAt === null) {
    lessonSeed.recordingStartedAt = meta.recordingStartedAt;
    if (video.lessonDate === null) {
      lessonSeed.lessonDate = localDateInSchoolTz(meta.recordingStartedAt, await schoolTimezone());
    }
  }

  await updateVideo(videoId, {
    durationMs: meta.durationMs,
    fps: meta.fps,
    width: meta.width,
    height: meta.height,
    ...(thumbnailPath ? { thumbnailPath } : {}),
    ...lessonSeed,
    status: "analyzing",
    progress: 0.05,
  });
}

/**
 * Board and door zones are NOT probed here any more.
 *
 * There used to be two steps at this point, each POSTing /detect-board and
 * /detect-door, and each of those re-scanned the entire video to place one
 * rectangle. That was close to free on a 4-minute clip and untenable on a real
 * lesson: iter_frames decodes every frame whatever the sample rate, so a zone
 * scan costs about what the analysis costs, and on a 37-minute video both calls
 * ran past RunPod's proxy timeout and came back 524 — ~4 minutes spent to
 * return nothing, after which the lesson was analysed with no zones at all and
 * reported board time as null.
 *
 * The analysis pass already detects Screen and Door on every sampled frame, so
 * it now places any missing zone from its own detections and uses it in the
 * same run (jobs.derive_result). ingestStep persists whatever it proposed.
 * POST /detect-board still exists for the zone editor's manual button.
 */
async function startAnalysisStep(
  videoId: string,
  attemptId: string | undefined,
  jobId: string,
): Promise<string> {
  const video = await requireCurrentRun(videoId, attemptId, jobId, "start-analysis");

  // Refuse a CPU pod before it costs anything. A pod whose driver is too old
  // for the cu13 torch build comes up RUNNING and answers /health perfectly —
  // it just resolves to device "cpu". REQUIRE_DEVICE does stop the run, but
  // only after the image pull, an 845 MB download and a checkpoint load; a
  // 37-minute lesson burned 7 minutes of billed GPU discovering it. The device
  // is one cheap GET away, so ask first.
  const health = await mlHealth();
  if (health && health.device !== "cuda") {
    throw new UnrecoverableError(
      `ML service resolved device "${health.device}", not cuda — the pod's driver is ` +
        "probably too old for this image's CUDA build. Recreate the pod; do not widen " +
        "the CUDA pin in Settings.",
    );
  }

  const zones = await getZones(videoId);
  const runTokens = [attemptId, jobId].filter((t): t is string => Boolean(t));
  return mlStartAnalysis({
    videoId,
    videoPath: mediaSource(video.filePath),
    sampleFps: 5,
    zones,
    period: periodOffsets(video, await schoolTimezone()),
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

  // Zones the analysis placed for itself. It has already USED them, so the
  // KPIs in `result` assume they exist — dropping them here would leave the
  // numbers unexplainable in the UI and force a re-proposal next run. Guarded
  // by hasZoneKind so a hand-drawn zone is never overwritten.
  for (const zone of result.proposed_zones ?? []) {
    if (await hasZoneKind(videoId, zone.kind)) continue;
    await insertZone(videoId, {
      kind: zone.kind,
      polygon: zone.polygon,
      meta: { auto: true, confidence: zone.confidence, method: zone.method },
    });
    logger.info(
      { videoId, kind: zone.kind, confidence: zone.confidence },
      "zone auto-placed from analysis",
    );
  }

  await replaceDerived(videoId, result, { markDone: true });
}

export async function processAnalyzeJob(job: Job<AnalyzeJobData>): Promise<void> {
  const { videoId, attemptId } = job.data;
  const jobId = String(job.id);
  try {
    await probeStep(videoId, attemptId, jobId);
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
