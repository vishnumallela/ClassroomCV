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
import { enqueueAnalysis } from "@api/lib/queue";
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
