import * as z from "zod";
import { getVideo, replaceZones } from "@api/db/queries";
import { applyRederive } from "@api/analysis/rederive";
import { base } from "@api/orpc/base";

const Point = z.tuple([z.number().min(0).max(1), z.number().min(0).max(1)]);
const ZoneSchema = z.object({
  kind: z.enum(["board", "door"]),
  polygon: z.array(Point).min(3).max(1000),
});

const UpsertInput = z.object({ id: z.string(), zones: z.array(ZoneSchema).max(8) });

export const zonesRouter = {
  upsert: base.input(UpsertInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    if (video.status !== "done" && video.status !== "failed") {
      throw errors.CONFLICT({ message: "Cannot edit zones during analysis." });
    }
    await replaceZones(input.id, input.zones);
    try {
      await applyRederive(input.id, input.zones, { markDone: false });
    } catch {
      throw errors.DEPENDENCY_UNAVAILABLE({ message: "Zones saved, but re-derivation failed." });
    }
    return { ok: true as const };
  }),
};
