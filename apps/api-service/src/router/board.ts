import * as z from "zod";
import { getVideo } from "@api/db/queries";
import { mlDetectBoard } from "@api/lib/ml";
import { base } from "@api/orpc/base";

export const boardRouter = {
  detect: base.input(z.object({ id: z.string() })).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    try {
      const res = await mlDetectBoard(input.id, video.filePath);
      return {
        polygon: res.polygon,
        confidence: res.confidence,
        method: res.method,
        frameTsMs: res.frame_ts_ms,
      };
    } catch {
      throw errors.DEPENDENCY_UNAVAILABLE({ message: "Board detection failed." });
    }
  }),
};
