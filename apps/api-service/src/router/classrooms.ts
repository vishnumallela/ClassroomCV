import * as z from "zod";
import {
  countClassroomVideos,
  createClassroom,
  deleteClassroomRow,
  getClassroom,
  getClassroomMetrics,
  getClassroomZones,
  getTimetable,
  listClassrooms,
  listVideos,
  replaceClassroomZones,
  replaceTimetable,
  updateClassroom,
} from "@api/db/queries";
import { base } from "@api/orpc/base";

const IdInput = z.object({ id: z.string() });
const Name = z.string().trim().min(1).max(120);
const OptionalText = z.string().trim().max(500).nullish();

const Point = z.tuple([z.number().min(0).max(1), z.number().min(0).max(1)]);
const ZoneSchema = z.object({
  kind: z.enum(["board", "door"]),
  polygon: z.array(Point).min(3).max(1000),
});

const TIME = /^\d{2}:\d{2}(:\d{2})?$/;
const TimetableRowSchema = z.object({
  weekday: z.number().int().min(1).max(7),
  slot: z.number().int().min(0).max(40),
  label: z.string().trim().min(1).max(60),
  scheduledStart: z.string().regex(TIME),
  scheduledEnd: z.string().regex(TIME),
  subject: z.string().trim().max(120).nullish(),
  teacher: z.string().trim().max(120).nullish(),
  yearGroup: z.string().trim().max(60).nullish(),
});

export const classroomsRouter = {
  list: base.handler(() => listClassrooms()),

  create: base
    .input(z.object({ name: Name, location: OptionalText, description: OptionalText }))
    .handler(async ({ input }) => {
      const row = await createClassroom({
        name: input.name,
        location: input.location ?? null,
        description: input.description ?? null,
      });
      return { id: row.id };
    }),

  get: base.input(IdInput).handler(async ({ input, errors }) => {
    const classroom = await getClassroom(input.id);
    if (!classroom) throw errors.NOT_FOUND();
    const [zones, videos, timetable] = await Promise.all([
      getClassroomZones(input.id),
      listVideos(input.id),
      getTimetable(input.id),
    ]);
    return {
      id: classroom.id,
      name: classroom.name,
      location: classroom.location,
      description: classroom.description,
      createdAt: classroom.createdAt.toISOString(),
      zones,
      videos,
      timetable: timetable.map((r) => ({
        id: r.id,
        weekday: r.weekday,
        slot: r.slot,
        label: r.label,
        scheduledStart: r.scheduledStart,
        scheduledEnd: r.scheduledEnd,
        subject: r.subject,
        teacher: r.teacher,
        yearGroup: r.yearGroup,
      })),
    };
  }),

  update: base
    .input(
      z.object({
        id: z.string(),
        name: Name.optional(),
        location: OptionalText,
        description: OptionalText,
      }),
    )
    .handler(async ({ input, errors }) => {
      const { id, ...patch } = input;
      const classroom = await getClassroom(id);
      if (!classroom) throw errors.NOT_FOUND();
      const set = {
        ...(patch.name !== undefined ? { name: patch.name } : {}),
        ...(patch.location !== undefined ? { location: patch.location } : {}),
        ...(patch.description !== undefined ? { description: patch.description } : {}),
      };
      // Drizzle's .set({}) throws; an id-only request is a valid no-op.
      if (Object.keys(set).length > 0) await updateClassroom(id, set);
      return { ok: true as const };
    }),

  delete: base.input(IdInput).handler(async ({ input, errors }) => {
    const classroom = await getClassroom(input.id);
    if (!classroom) throw errors.NOT_FOUND();
    const videoCount = await countClassroomVideos(input.id);
    if (videoCount > 0) {
      throw errors.CONFLICT({
        message: `Classroom still holds ${videoCount} lesson${videoCount === 1 ? "" : "s"}. Delete them first.`,
      });
    }
    try {
      await deleteClassroomRow(input.id);
    } catch (err) {
      // An upload can insert a video between the count and the delete; the
      // restrict FK (23503) protects the data — surface the same CONFLICT
      // instead of a raw 500.
      if ((err as { code?: string })?.code === "23503") {
        throw errors.CONFLICT({
          message: "Classroom still holds lessons. Delete them first.",
        });
      }
      throw err;
    }
    return { ok: true as const };
  }),

  // Replace the classroom's zone template. Applies to lessons uploaded from
  // now on (each new upload is seeded from the template); already-analyzed
  // lessons keep their own zones and are re-derived per video if edited there.
  setZones: base
    .input(z.object({ id: z.string(), zones: z.array(ZoneSchema).max(8) }))
    .handler(async ({ input, errors }) => {
      const classroom = await getClassroom(input.id);
      if (!classroom) throw errors.NOT_FOUND();
      await replaceClassroomZones(input.id, input.zones);
      return { ok: true as const };
    }),

  // Replace the classroom's week. Lessons resolve their bells from it at read
  // time (lib/timetable.ts), so a correction here reaches every lesson in the
  // room that has no bells of its own, with no re-analysis.
  setTimetable: base
    .input(z.object({ id: z.string(), rows: z.array(TimetableRowSchema).max(80) }))
    .handler(async ({ input, errors }) => {
      const classroom = await getClassroom(input.id);
      if (!classroom) throw errors.NOT_FOUND();
      const seen = new Set<string>();
      for (const r of input.rows) {
        if (r.scheduledEnd <= r.scheduledStart) {
          throw errors.VALIDATION({
            message: `${r.label}: scheduled end must be after scheduled start.`,
          });
        }
        const key = `${r.weekday}:${r.slot}`;
        if (seen.has(key)) {
          throw errors.VALIDATION({ message: `Two periods share slot ${r.slot} on the same day.` });
        }
        seen.add(key);
      }
      await replaceTimetable(
        input.id,
        input.rows.map((r) => ({
          weekday: r.weekday,
          slot: r.slot,
          label: r.label,
          scheduledStart: r.scheduledStart,
          scheduledEnd: r.scheduledEnd,
          subject: r.subject?.trim() || null,
          teacher: r.teacher?.trim() || null,
          yearGroup: r.yearGroup?.trim() || null,
        })),
      );
      return { ok: true as const };
    }),

  metrics: base.input(IdInput).handler(async ({ input, errors }) => {
    const classroom = await getClassroom(input.id);
    if (!classroom) throw errors.NOT_FOUND();
    return getClassroomMetrics(input.id);
  }),
};
