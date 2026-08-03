import * as z from "zod";
import {
  countClassroomVideos,
  createClassroom,
  deleteClassroomRow,
  getClassroom,
  getClassroomMetrics,
  getClassroomZones,
  listClassrooms,
  listVideos,
  replaceClassroomZones,
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
    const [zones, videos] = await Promise.all([
      getClassroomZones(input.id),
      listVideos(input.id),
    ]);
    return {
      id: classroom.id,
      name: classroom.name,
      location: classroom.location,
      description: classroom.description,
      createdAt: classroom.createdAt.toISOString(),
      zones,
      videos,
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
      await updateClassroom(id, {
        ...(patch.name !== undefined ? { name: patch.name } : {}),
        ...(patch.location !== undefined ? { location: patch.location } : {}),
        ...(patch.description !== undefined ? { description: patch.description } : {}),
      });
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
    await deleteClassroomRow(input.id);
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

  metrics: base.input(IdInput).handler(async ({ input, errors }) => {
    const classroom = await getClassroom(input.id);
    if (!classroom) throw errors.NOT_FOUND();
    return getClassroomMetrics(input.id);
  }),
};
