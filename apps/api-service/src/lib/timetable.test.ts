import { describe, expect, test } from "bun:test";
import { isoWeekday, normaliseTime, resolveSchedule, type TimetableRow } from "@api/lib/timetable";

const IST = "Asia/Kolkata";

// The 7B sheet (Khaitan Public School, 2026): periods are NOT back-to-back.
function day(weekday: number): TimetableRow[] {
  const bells: [string, string, string][] = [
    ["C.T", "07:45", "07:55"],
    ["Period 1", "07:55", "08:40"],
    ["Period 2", "08:40", "09:25"],
    ["Period 3", "09:50", "10:35"],
    ["Period 4", "10:35", "11:20"],
    ["Period 5", "11:20", "12:05"],
    ["Period 6", "12:05", "12:45"],
    ["Period 7", "12:55", "13:35"],
    ["Period 8", "13:35", "14:15"],
  ];
  return bells.map(([label, scheduledStart, scheduledEnd], slot) => ({
    weekday,
    slot,
    label,
    scheduledStart: `${scheduledStart}:00`,
    scheduledEnd: `${scheduledEnd}:00`,
    subject: label === "Period 3" ? "english" : null,
    teacher: label === "Period 3" ? "Ms A" : null,
    yearGroup: "class 7",
  }));
}

const MONDAY = day(1);

// The real handover recording: 2026-08-17 (a Monday), first frame 09:49:58.
const video = {
  recordingStartedAt: new Date("2026-08-17T04:19:58Z"),
  durationMs: 2_700_922,
  lessonDate: "2026-08-17",
  period: null as string | null,
  scheduledStart: null as string | null,
  scheduledEnd: null as string | null,
  subject: null as string | null,
  yearGroup: null as string | null,
  hasFollowingPeriod: null as boolean | null,
};

describe("isoWeekday", () => {
  test("Monday is 1, Sunday is 7", () => {
    expect(isoWeekday("2026-08-17")).toBe(1);
    expect(isoWeekday("2026-08-23")).toBe(7);
    expect(isoWeekday("not a date")).toBeNull();
  });
});

describe("normaliseTime", () => {
  test("accepts the shapes the DB and the form produce", () => {
    expect(normaliseTime("9:50")).toBe("09:50:00");
    expect(normaliseTime("09:50")).toBe("09:50:00");
    expect(normaliseTime("09:50:00")).toBe("09:50:00");
    expect(normaliseTime("")).toBeNull();
  });
});

describe("resolveSchedule", () => {
  test("a recording with no bells of its own takes them from the period it overlaps most", () => {
    const s = resolveSchedule(video, MONDAY, IST);
    expect(s.source).toBe("timetable");
    expect(s.weekday).toBe(1);
    expect(s.period).toBe("Period 3");
    expect(s.scheduledStart).toBe("09:50:00");
    expect(s.scheduledEnd).toBe("10:35:00");
    expect(s.subject).toBe("english");
    expect(s.teacher).toBe("Ms A");
  });

  test("the period label typed on the video wins over the overlap, however it is spelled", () => {
    for (const label of ["Period 4", "period 4", "P4", "4"]) {
      const s = resolveSchedule({ ...video, period: label }, MONDAY, IST);
      expect(s.scheduledStart).toBe("10:35:00");
      expect(s.period).toBe(label);
    }
  });

  test("bells typed on the video override the timetable, and still place the lesson in it", () => {
    const s = resolveSchedule(
      { ...video, period: "Period 3", scheduledStart: "09:52", scheduledEnd: "10:35:00" },
      MONDAY,
      IST,
    );
    expect(s.source).toBe("video");
    expect(s.scheduledStart).toBe("09:52:00");
    // the previous period still comes from the table
    expect(s.previousPeriod).toEqual({ label: "Period 2", scheduledEnd: "09:25:00" });
  });

  test("a following period is one that starts exactly where this one ends", () => {
    expect(resolveSchedule({ ...video, period: "Period 3" }, MONDAY, IST).hasFollowingPeriod).toBe(
      true,
    ); // period 4 at 10:35
    expect(resolveSchedule({ ...video, period: "Period 2" }, MONDAY, IST).hasFollowingPeriod).toBe(
      false,
    ); // a 25-minute break
    expect(resolveSchedule({ ...video, period: "Period 8" }, MONDAY, IST).hasFollowingPeriod).toBe(
      false,
    );
  });

  test("the previous period is the latest one ending before this one starts", () => {
    const s = resolveSchedule({ ...video, period: "Period 3" }, MONDAY, IST);
    expect(s.previousPeriod).toEqual({ label: "Period 2", scheduledEnd: "09:25:00" });
    expect(resolveSchedule({ ...video, period: "C.T" }, MONDAY, IST).previousPeriod).toBeNull();
  });

  test("the lesson's weekday comes from the date, or from the anchor in the school timezone", () => {
    // no timetable rows for a Tuesday: nothing resolves
    const tuesday = resolveSchedule({ ...video, lessonDate: "2026-08-18" }, MONDAY, IST);
    expect(tuesday.weekday).toBe(2);
    expect(tuesday.source).toBeNull();
    // 2026-08-16T20:30Z is already Monday 02:00 in Kolkata
    const byAnchor = resolveSchedule(
      {
        ...video,
        lessonDate: null,
        recordingStartedAt: new Date("2026-08-16T20:30:00Z"),
        durationMs: null,
      },
      MONDAY,
      IST,
    );
    expect(byAnchor.weekday).toBe(1);
  });

  test("without a timetable the video's own fields stand, and nothing is invented", () => {
    const s = resolveSchedule({ ...video, period: "Period 3", subject: "maths" }, [], IST);
    expect(s.source).toBeNull();
    expect(s.scheduledStart).toBeNull();
    expect(s.period).toBe("Period 3");
    expect(s.subject).toBe("maths");
    expect(s.previousPeriod).toBeNull();
    expect(s.hasFollowingPeriod).toBeNull();
  });

  test("a recording overlapping no period resolves nothing", () => {
    const s = resolveSchedule(
      { ...video, recordingStartedAt: new Date("2026-08-17T12:00:00Z"), durationMs: 60_000 }, // 17:30 IST
      MONDAY,
      IST,
    );
    expect(s.source).toBeNull();
    expect(s.scheduledStart).toBeNull();
  });
});
