import { describe, expect, test } from "bun:test";
import {
  lessonClock,
  localDateInSchoolTz,
  localTimeInSchoolTz,
  minutesAgainstSchedule,
  offsetToInstant,
  schoolTimeToInstant,
} from "@api/lib/school-time";

const IST = "Asia/Kolkata";

describe("schoolTimeToInstant", () => {
  test("reads a local wall clock as the right UTC instant", () => {
    // 11:15 IST is 05:45 UTC. India is UTC+5:30 year round.
    expect(schoolTimeToInstant("2026-07-08", "11:15", IST)?.toISOString()).toBe(
      "2026-07-08T05:45:00.000Z",
    );
  });

  test("accepts seconds, which is what Postgres `time` gives back", () => {
    expect(schoolTimeToInstant("2026-07-08", "11:15:00", IST)?.toISOString()).toBe(
      "2026-07-08T05:45:00.000Z",
    );
  });

  test("crosses a date boundary correctly for a half-hour offset zone", () => {
    // 00:30 IST is 19:00 UTC the PREVIOUS day — the case a naive
    // hours-only offset gets wrong.
    expect(schoolTimeToInstant("2026-07-08", "00:30", IST)?.toISOString()).toBe(
      "2026-07-07T19:00:00.000Z",
    );
  });

  test("handles a DST zone on both sides of the change", () => {
    // London is UTC+1 in July, UTC+0 in January. Same wall clock, different instants.
    expect(schoolTimeToInstant("2026-07-08", "09:00", "Europe/London")?.toISOString()).toBe(
      "2026-07-08T08:00:00.000Z",
    );
    expect(schoolTimeToInstant("2026-01-08", "09:00", "Europe/London")?.toISOString()).toBe(
      "2026-01-08T09:00:00.000Z",
    );
  });

  test("rejects malformed input rather than inventing a time", () => {
    expect(schoolTimeToInstant("08-07-2026", "11:15", IST)).toBeNull();
    expect(schoolTimeToInstant("2026-07-08", "11.15", IST)).toBeNull();
  });
});

describe("localDateInSchoolTz", () => {
  test("uses the school's calendar day, not UTC's", () => {
    // 19:30 UTC is already the next day in IST (01:00).
    expect(localDateInSchoolTz(new Date("2026-07-07T19:30:00Z"), IST)).toBe("2026-07-08");
  });

  test("formats as YYYY-MM-DD for a Postgres date column", () => {
    expect(localDateInSchoolTz(new Date("2026-07-08T05:45:00Z"), IST)).toBe("2026-07-08");
  });
});

describe("localTimeInSchoolTz", () => {
  test("renders the wall clock a person in the room would have seen", () => {
    // One of the real sample recordings: creation_time 2026-07-08T05:36:47Z.
    expect(localTimeInSchoolTz(new Date("2026-07-08T05:36:47Z"), IST)).toBe("11:06");
  });
});

describe("lessonClock", () => {
  const video = {
    recordingStartedAt: new Date("2026-07-08T05:36:47Z"),
    lessonDate: "2026-07-08",
    scheduledStart: "11:15:00",
    scheduledEnd: "12:00:00",
  };

  test("resolves both scheduled bounds to instants", () => {
    const clock = lessonClock(video, IST);
    expect(clock?.scheduledStartAt?.toISOString()).toBe("2026-07-08T05:45:00.000Z");
    expect(clock?.scheduledEndAt?.toISOString()).toBe("2026-07-08T06:30:00.000Z");
  });

  test("falls back to the recording's own date when none was entered", () => {
    const clock = lessonClock({ ...video, lessonDate: null }, IST);
    expect(clock?.scheduledStartAt?.toISOString()).toBe("2026-07-08T05:45:00.000Z");
  });

  test("is null without a recording anchor — Not Observed, never zero", () => {
    expect(lessonClock({ ...video, recordingStartedAt: null }, IST)).toBeNull();
  });

  test("leaves scheduled bounds null when the timetable is empty", () => {
    const clock = lessonClock({ ...video, scheduledStart: null, scheduledEnd: null }, IST);
    expect(clock).not.toBeNull();
    expect(clock?.scheduledStartAt).toBeNull();
    expect(clock?.scheduledEndAt).toBeNull();
  });
});

describe("minutesAgainstSchedule", () => {
  const clock = lessonClock(
    {
      // Recording rolls at 11:06 IST, nine minutes before the bell.
      recordingStartedAt: new Date("2026-07-08T05:36:47Z"),
      lessonDate: "2026-07-08",
      scheduledStart: "11:15:00",
      scheduledEnd: "12:00:00",
    },
    IST,
  )!;

  test("counts a teacher who appears after the bell as late", () => {
    // First seen 15 min in = 11:21:47 IST, which is 6m47s past an 11:15 bell.
    expect(minutesAgainstSchedule(clock, 15 * 60_000, clock.scheduledStartAt)).toBe(6.8);
  });

  test("counts a teacher already in the room as early, with a negative", () => {
    // First seen 2 min in = 11:08:47 IST, 6m13s before the bell.
    expect(minutesAgainstSchedule(clock, 2 * 60_000, clock.scheduledStartAt)).toBe(-6.2);
  });

  test("is null with no scheduled time, rather than reporting on time", () => {
    expect(minutesAgainstSchedule(clock, 15 * 60_000, null)).toBeNull();
  });
});

describe("offsetToInstant", () => {
  test("places a video offset on the wall clock", () => {
    const at = offsetToInstant(new Date("2026-07-08T05:36:47Z"), 15 * 60_000);
    expect(localTimeInSchoolTz(at, IST)).toBe("11:21");
  });
});
