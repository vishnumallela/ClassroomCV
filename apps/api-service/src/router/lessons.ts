import * as z from "zod";
import { getClassroom, getTimetable, getVideoDetail, listClassroomLessons } from "@api/db/queries";
import { base } from "@api/orpc/base";
import { buildDay, type RegisterVideo } from "@api/lib/register";
import { localDateInSchoolTz, schoolTimezone } from "@api/lib/school-time";
import { isoWeekday, resolveSchedule } from "@api/lib/timetable";
import { toDetailDto } from "@api/router/dto";

const DATE = /^\d{4}-\d{2}-\d{2}$/;

/**
 * The register: a classroom's day as periods, each assembled from the files
 * that cover it and from the next period's file where this period's teacher
 * was seen finishing (lib/register.ts).
 */
export const lessonsRouter = {
  day: base
    .input(z.object({ classroomId: z.string(), date: z.string().regex(DATE).optional() }))
    .handler(async ({ input, errors }) => {
      const classroom = await getClassroom(input.classroomId);
      if (!classroom) throw errors.NOT_FOUND();
      const tz = await schoolTimezone();
      const [timetable, lessons] = await Promise.all([
        getTimetable(input.classroomId),
        listClassroomLessons(input.classroomId),
      ]);

      // Which day each file belongs to: the typed lesson date, else the day
      // its recording started on in the school's timezone.
      const dated = lessons.map((v) => ({
        ...v,
        date:
          v.lessonDate ??
          (v.recordingStartedAt ? localDateInSchoolTz(v.recordingStartedAt, tz) : null),
      }));
      const dates = [...new Set(dated.map((v) => v.date).filter((d): d is string => d !== null))]
        .sort()
        .reverse();
      const date = input.date ?? dates[0] ?? null;
      if (!date)
        return {
          classroomId: classroom.id,
          date: null,
          dates,
          weekday: null,
          rows: [],
          undated: dated.length,
        };

      const weekday = isoWeekday(date);
      const periods = timetable.filter((r) => r.weekday === weekday);
      const todays = dated.filter((v) => v.date === date);

      const videos: RegisterVideo[] = await Promise.all(
        todays.map(async (v) => {
          const schedule = resolveSchedule(v, timetable, tz);
          if (v.status !== "done") {
            return {
              id: v.id,
              title: v.title,
              status: v.status,
              schedule,
              punctuality: null,
              arc: null,
              previousTeacher: null,
            };
          }
          const detail = await getVideoDetail(v.id);
          if (!detail) {
            return {
              id: v.id,
              title: v.title,
              status: v.status,
              schedule,
              punctuality: null,
              arc: null,
              previousTeacher: null,
            };
          }
          const dto = toDetailDto(detail, tz);
          return {
            id: v.id,
            title: v.title,
            status: v.status,
            schedule,
            punctuality: dto.punctuality,
            arc: dto.arc,
            previousTeacher: dto.previousTeacher,
          };
        }),
      );

      return {
        classroomId: classroom.id,
        date,
        dates,
        weekday,
        rows: buildDay(periods, videos),
        // Files that could not be placed on any day: no date typed and no anchor.
        undated: dated.filter((v) => v.date === null).length,
      };
    }),
};
