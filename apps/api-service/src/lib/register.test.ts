import { describe, expect, test } from "bun:test";
import { buildDay, type RegisterVideo } from "@api/lib/register";
import type { ResolvedSchedule, TimetableRow } from "@api/lib/timetable";

const periods: TimetableRow[] = [
  {
    weekday: 1,
    slot: 2,
    label: "Period 2",
    scheduledStart: "08:40:00",
    scheduledEnd: "09:25:00",
    subject: "maths",
    teacher: "Ms P",
    yearGroup: "class 7",
  },
  {
    weekday: 1,
    slot: 3,
    label: "Period 3",
    scheduledStart: "09:50:00",
    scheduledEnd: "10:35:00",
    subject: "english",
    teacher: null,
    yearGroup: "class 7",
  },
  {
    weekday: 1,
    slot: 4,
    label: "Period 4",
    scheduledStart: "10:35:00",
    scheduledEnd: "11:20:00",
    subject: null,
    teacher: null,
    yearGroup: "class 7",
  },
];

function schedule(period: string | null): ResolvedSchedule {
  return {
    weekday: 1,
    period,
    scheduledStart: null,
    scheduledEnd: null,
    subject: null,
    yearGroup: null,
    teacher: null,
    hasFollowingPeriod: null,
    previousPeriod: null,
    source: period ? "timetable" : null,
  };
}

// The real day: only the period-3 file exists. Its teacher arrived 4.1 min
// late and left on the bell; the period-2 teacher was seen leaving at 09:55.
const period3File: RegisterVideo = {
  id: "v3",
  title: "period 3 file",
  status: "done",
  schedule: schedule("Period 3"),
  punctuality: {
    arrivalAt: "09:54",
    departureAt: "10:34",
    arrivalMinutesLate: 4.1,
    departureMinutesLate: 0,
    presenceShareOfPeriod: 0.876,
    notObservedReason: null,
  },
  arc: {
    start: { value: 289_000, state: "provisional" },
    end: { value: 2_694_000, state: "provisional" },
    startDelayMin: { value: 4.8 },
    overrunMin: { value: -0.1 },
  },
  previousTeacher: {
    state: "observed",
    departureAt: "09:55",
    departureMinutesIntoPeriod: 5.6,
    previousPeriodLabel: "Period 2",
    previousPeriodEnd: "09:25",
    departureMinutesAfterHerBell: 30.6,
    breakMinutesBeforeThisPeriod: 25,
    presenceMsInThisFile: 320_000,
    boardMsInThisFile: 80_000,
  },
};

describe("buildDay", () => {
  test("period 3 is its own teacher's, and only hers", () => {
    const rows = buildDay(periods, [period3File]);
    const p3 = rows.find((r) => r.label === "Period 3")!;
    expect(p3.covering.map((c) => c.id)).toEqual(["v3"]);
    expect(p3.own?.arrivalMinutesLate).toBe(4.1);
    expect(p3.arrival.state).toBe("observed");
    expect(p3.departure).toEqual({ state: "observed", reason: null, from: "own" });
    expect(p3.overrunMin).toBe(0);
    expect(p3.spillover).toBeNull();
  });

  test("period 2's departure and over-run come from the period-3 file, with no file of its own", () => {
    const rows = buildDay(periods, [period3File]);
    const p2 = rows.find((r) => r.label === "Period 2")!;
    expect(p2.covering).toEqual([]);
    expect(p2.own).toBeNull();
    expect(p2.arrival.state).toBe("not_observed");
    expect(p2.arrival.reason).toBe("No recording covers this period.");
    expect(p2.departure.state).toBe("observed");
    expect(p2.departure.from).toBe("spillover");
    expect(p2.spillover?.departureAt).toBe("09:55");
    expect(p2.spillover?.presenceMs).toBe(320_000);
    expect(p2.spillover?.boardMs).toBe(80_000);
    // she stayed 30.6 min past her own 09:25 bell, 25 of them the break
    expect(p2.overrunMin).toBe(30.6);
    expect(p2.spillover?.breakMinutes).toBe(25);
    expect(p2.teacher).toBe("Ms P");
  });

  test("a period with no file and no spill-over is Not Observed on both counts", () => {
    const rows = buildDay(periods, [period3File]);
    const p4 = rows.find((r) => r.label === "Period 4")!;
    expect(p4.arrival.state).toBe("not_observed");
    expect(p4.departure.state).toBe("not_observed");
    expect(p4.overrunMin).toBeNull();
  });

  test("a file still analysing covers its period but observes nothing yet", () => {
    const rows = buildDay(periods, [
      { ...period3File, status: "analyzing", punctuality: null, arc: null, previousTeacher: null },
    ]);
    const p3 = rows.find((r) => r.label === "Period 3")!;
    expect(p3.covering).toHaveLength(1);
    expect(p3.own).toBeNull();
    expect(p3.arrival.reason).toBe("The recording is not analysed yet.");
  });
});
