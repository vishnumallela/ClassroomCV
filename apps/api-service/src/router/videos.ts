import { rm } from "node:fs/promises";
import { dirname } from "node:path";
import * as z from "zod";
import {
  deleteVideoRows,
  getDetections,
  getVideo,
  getVideoDetail,
  getVideoStatus,
  listVideos,
} from "@api/db/queries";
import { removeObjects } from "@api/lib/storage";
import { base } from "@api/orpc/base";
import { toDetailDto } from "@api/router/dto";

const IdInput = z.object({ id: z.string() });

export const videosRouter = {
  list: base.handler(() => listVideos()),

  get: base.input(IdInput).handler(async ({ input, errors }) => {
    const detail = await getVideoDetail(input.id);
    if (!detail) throw errors.NOT_FOUND();
    return toDetailDto(detail);
  }),

  status: base.input(IdInput).handler(async ({ input, errors }) => {
    const status = await getVideoStatus(input.id);
    if (!status) throw errors.NOT_FOUND();
    return status;
  }),

  detections: base
    .input(z.object({ id: z.string(), fps: z.number().positive().optional() }))
    .handler(({ input }) => getDetections(input.id, input.fps)),

  delete: base.input(IdInput).handler(async ({ input, errors }) => {
    const video = await getVideo(input.id);
    if (!video) throw errors.NOT_FOUND();
    await deleteVideoRows(input.id);
    await rm(dirname(video.filePath), { recursive: true, force: true }).catch(() => undefined);
    await removeObjects([video.filePath, video.thumbnailPath ?? ""]).catch(() => undefined);
    return { ok: true as const };
  }),
};
