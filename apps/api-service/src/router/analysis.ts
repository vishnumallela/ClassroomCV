import * as z from "zod";
import {
  countDetectionEvents,
  getVideo,
  getVideoDetail,
  setWorkflowRunId,
  updateStatus,
  wipeDerived,
} from "@api/db/queries";
import { rederiveFromRaw } from "@api/analysis/rederive";
import { enqueueAnalysis, enqueueRederive } from "@api/lib/queue";
import { base } from "@api/orpc/base";
import { toDetailDto } from "@api/router/dto";

const IdInput = z.object({ id: z.string() });

export const analysisRouter = {
  reanalyze: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();

    // "deriving" counts as eligible too. Detections already exist by then, so a
    // re-derive is always the right move -- and if a previous derive was
    // interrupted the video is STUCK in this state, where treating it as
    // unsettled silently escalates to a full YOLO re-run and wipes the derived
    // tables. That is a very expensive answer to "try that again".
    const settled =
      video.status === "done" || video.status === "failed" || video.status === "deriving";
    if (settled && (await countDetectionEvents(input.id)) > 0) {
      // Queued, not awaited. Deriving a 37-minute lesson takes minutes, which
      // outlives the HTTP request: it used to die at 300s and report
      // "Re-derivation failed" while the derive was still running happily.
      // The client watches video.status, the same way it does for an analysis.
      await updateStatus(input.id, { status: "deriving", progress: 0.1, error: null });
      await enqueueRederive(input.id);
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

  rederive: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    if (video.status !== "done" && video.status !== "failed") {
      throw errors.CONFLICT({ message: "Cannot rederive during analysis." });
    }
    try {
      await rederiveFromRaw(input.id);
    } catch {
      throw errors.DEPENDENCY_UNAVAILABLE({ message: "Re-derivation failed." });
    }
    const detail = await getVideoDetail(input.id);
    return toDetailDto(detail!);
  }),
};
