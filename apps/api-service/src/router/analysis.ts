import * as z from "zod";
import {
  countDetectionEvents,
  getVideo,
  getVideoDetail,
  setAudioStatus,
  setWorkflowRunId,
  updateStatus,
  wipeDerived,
} from "@api/db/queries";
import { rederiveFromRaw } from "@api/analysis/rederive";
import { enqueueAnalysis, enqueueAudioAnalysis } from "@api/lib/queue";
import { schoolTimezone } from "@api/lib/school-time";
import { base } from "@api/orpc/base";
import { toDetailDto } from "@api/router/dto";

const IdInput = z.object({ id: z.string() });

export const analysisRouter = {
  reanalyze: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();

    const settled = video.status === "done" || video.status === "failed";
    if (settled && (await countDetectionEvents(input.id)) > 0) {
      try {
        await rederiveFromRaw(input.id);
      } catch {
        throw errors.DEPENDENCY_UNAVAILABLE({ message: "Re-derivation failed." });
      }
      return { ok: true as const, mode: "rederived" as const };
    }

    // Full restart. Set the fence token before enqueue so any in-flight job is
    // superseded on its next fence check.
    const attemptId = crypto.randomUUID();
    await setWorkflowRunId(input.id, attemptId);
    await wipeDerived(input.id);
    await updateStatus(input.id, { status: "queued", progress: 0, error: null });
    await enqueueAnalysis({ videoId: input.id, attemptId });
    return { ok: true as const, mode: "restarted" as const };
  }),

  // Re-run the audio half. The stored transcript id is reused, so this
  // re-cuts and re-stores the sentences without paying to transcribe again;
  // clearing the id first (not offered here) is what a real re-transcription
  // would need.
  reanalyzeAudio: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    if (video.audioStatus === "extracting" || video.audioStatus === "transcribing") {
      throw errors.CONFLICT({ message: "Audio analysis is already running." });
    }
    await setAudioStatus(input.id, { audioStatus: "queued", audioError: null });
    await enqueueAudioAnalysis({ videoId: input.id });
    return { ok: true as const };
  }),

  rederive: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    if (video.status !== "done" && video.status !== "failed") {
      throw errors.CONFLICT({ message: "Cannot rederive during analysis." });
    }
    // Raw detections age out (2-day hot-tier retention). Rederiving from an
    // empty hot tier would REPLACE the stored analytics and the permanent
    // overlay keyframes with zeros — irrecoverably, short of a full YOLO
    // re-run. Refuse instead of destroying.
    if ((await countDetectionEvents(input.id)) === 0) {
      throw errors.CONFLICT({
        message:
          "Raw detections for this lesson have aged out; use Re-analyze to run the full pipeline again.",
      });
    }
    try {
      await rederiveFromRaw(input.id);
    } catch {
      throw errors.DEPENDENCY_UNAVAILABLE({ message: "Re-derivation failed." });
    }
    const detail = await getVideoDetail(input.id);
    return toDetailDto(detail!, await schoolTimezone());
  }),
};
