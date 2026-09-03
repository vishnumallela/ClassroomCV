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
  updateLessonDetails,
} from "@api/db/queries";
import { schoolTimezone } from "@api/lib/school-time";
import { removeObjects } from "@api/lib/storage";
import { base } from "@api/orpc/base";
import { toDetailDto } from "@api/router/dto";

const IdInput = z.object({ id: z.string() });

/**
 * The lesson-details form (docs/teacher-measurements.md, P1-P5).
 *
 * Everything is optional and nullable: an upload is never blocked on these, and
 * a person filling in half the form must not have the other half cleared.
 * `null` is an explicit erase; a missing key leaves the column alone.
 */
const TIME = /^\d{2}:\d{2}(:\d{2})?$/;
const DATE = /^\d{4}-\d{2}-\d{2}$/;

const LessonDetailsInput = z.object({
  id: z.string(),
  // ISO instant. Read from the container's creation_time at probe time; typed
  // in only for the recordings that carry no tag.
  recordingStartedAt: z.iso.datetime().nullish(),
  lessonDate: z.string().regex(DATE).nullish(),
  period: z.string().trim().max(60).nullish(),
  scheduledStart: z.string().regex(TIME).nullish(),
  scheduledEnd: z.string().regex(TIME).nullish(),
  subject: z.string().trim().max(120).nullish(),
  yearGroup: z.string().trim().max(60).nullish(),
  roomType: z.enum(["classroom", "lab", "pe", "library", "practical"]).nullish(),
  hasFollowingPeriod: z.boolean().nullish(),
});

export const videosRouter = {
  list: base
    .input(z.object({ classroomId: z.string().optional() }).optional())
    .handler(({ input }) => listVideos(input?.classroomId)),

  get: base.input(IdInput).handler(async ({ input, errors }) => {
    const detail = await getVideoDetail(input.id);
    if (!detail) throw errors.NOT_FOUND();
    return toDetailDto(detail, await schoolTimezone());
  }),

  setLessonDetails: base.input(LessonDetailsInput).handler(async ({ input, errors }) => {
    const { id, recordingStartedAt, ...rest } = input;
    if (!(await getVideo(id))) throw errors.NOT_FOUND();

    // A scheduled end before its start is almost always a typo (13:00-12:00),
    // and it would make the period length negative — which silently poisons
    // R5's share and R12's overrun rather than failing loudly.
    if (rest.scheduledStart && rest.scheduledEnd && rest.scheduledEnd <= rest.scheduledStart) {
      throw errors.VALIDATION({ message: "Scheduled end must be after scheduled start." });
    }

    const updated = await updateLessonDetails(id, {
      ...rest,
      ...(recordingStartedAt === undefined
        ? {}
        : { recordingStartedAt: recordingStartedAt ? new Date(recordingStartedAt) : null }),
    });
    if (!updated) throw errors.NOT_FOUND();

    const detail = await getVideoDetail(id);
    if (!detail) throw errors.NOT_FOUND();
    return toDetailDto(detail, await schoolTimezone());
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
