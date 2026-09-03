import { describe, expect, test } from "bun:test";
import type { DataQuality } from "@api/db/schema";
import { toDetailDto } from "@api/router/dto";

const IST = "Asia/Kolkata";

/**
 * A lesson recorded at 09:49:58 for a period scheduled 09:50–10:35, with the
 * teacher present from the first frame.
 *
 * These are the numbers from the recording that prompted the multi-adult work:
 * on it, "present from the first frame" is the OUTGOING teacher, and reporting
 * her arrival as this lesson's makes a teacher who was ~4.5 minutes late read
 * as on time.
 */
function detail(dataQuality: DataQuality | null, presenceMs = 2_700_000) {
  return {
    video: {
      id: "v1",
      classroomId: "c1",
      title: "Period 3",
      originalFilename: "lesson.mp4",
      durationMs: 2_700_000,
      fps: 25,
      width: 1920,
      height: 1080,
      status: "done",
      progress: 1,
      error: null,
      thumbnailPath: null,
      uploadedAt: new Date("2026-08-17T04:20:00.000Z"),
      recordingStartedAt: new Date("2026-08-17T04:19:58.000Z"), // 09:49:58 IST
      lessonDate: "2026-08-17",
      period: "Period 3",
      scheduledStart: "09:50:00",
      scheduledEnd: "10:35:00",
      subject: "Biology",
      yearGroup: "Class 12",
      roomType: "classroom",
      hasFollowingPeriod: null,
    },
    classroom: null,
    zones: [],
    tracks: [],
    events: [],
    analytics: {
      teacherPresentMs: presenceMs,
      teacherBoardMs: null,
      teacherPointingMs: null,
      teacherWritingMs: null,
      entries: 1,
      exits: 0,
      presenceIntervals: [[0, presenceMs]],
      boardIntervals: [],
      pointingIntervals: [],
      writingIntervals: [],
      entryExit: [],
      heatmap: { grid_w: 0, grid_h: 0, teacher: [] },
      dataQuality,
    },
    // The DTO reads a narrow slice of the query result; the rest of the row
    // shape is irrelevant to punctuality and is not worth reproducing here.
  } as unknown as Parameters<typeof toDetailDto>[0];
}

function quality(over: Partial<DataQuality> = {}): DataQuality {
  return {
    detections: 1000,
    frames: 1000,
    sampled_frames: 1000,
    coverage: 1,
    mean_confidence: 0.8,
    breaks: 0,
    longest_gap_ms: 0,
    confidence: {
      overall: "high",
      coverage: "high",
      continuity: "high",
      teacher: "high",
      attribution: "high",
    },
    notes: [],
    ...over,
  };
}

describe("toPunctuality", () => {
  test("reports arrival and departure for a one-adult lesson", () => {
    const { punctuality } = toDetailDto(detail(quality()), IST);
    expect(punctuality.arrivalAt).toBe("09:49");
    // toBeCloseTo, not toBe: the recording starts 2s before the bell, so this
    // rounds to -0 and `toBe(0)` fails on the sign. The UI reads anything under
    // half a minute as "on time" either way.
    expect(punctuality.arrivalMinutesLate).toBeCloseTo(0);
    expect(punctuality.notObservedReason).toBeNull();
  });

  test("withholds every Group A number when two adults were in the room", () => {
    const blended = quality({
      multiple_adults_detected: true,
      max_simultaneous_adults: 2,
      co_presence_ms: 58_000,
      confidence: {
        overall: "low",
        coverage: "high",
        continuity: "high",
        teacher: "high",
        attribution: "low",
      },
    });
    const { punctuality } = toDetailDto(detail(blended), IST);

    // R1-R5, all of them. Reporting any one of these would be reporting it for
    // whichever adult the detector scored highest frame by frame.
    expect(punctuality.arrivalAt).toBeNull();
    expect(punctuality.departureAt).toBeNull();
    expect(punctuality.arrivalMinutesLate).toBeNull();
    expect(punctuality.departureMinutesLate).toBeNull();
    expect(punctuality.presenceShareOfPeriod).toBeNull();
  });

  test("says WHY it withheld them", () => {
    const blended = quality({ multiple_adults_detected: true });
    const { punctuality } = toDetailDto(detail(blended), IST);

    // Without this the refusal is indistinguishable from an unfilled timetable
    // field, and gets "fixed" by typing the bell times in again.
    expect(punctuality.notObservedReason).toContain("More than one adult");
    expect(punctuality.recordingStartedAt).not.toBeNull();
  });

  test("a lesson analysed before the check keeps its numbers", () => {
    // The field is absent, which means nobody looked — not that one adult was
    // measured. Withholding here would retroactively blank the whole archive on
    // evidence that was never gathered.
    const older = quality();
    delete (older as Partial<DataQuality>).multiple_adults_detected;
    const { punctuality } = toDetailDto(detail(older), IST);

    expect(punctuality.arrivalAt).toBe("09:49");
    expect(punctuality.notObservedReason).toBeNull();
  });

  test("a lesson with no quality report at all keeps its numbers", () => {
    const { punctuality } = toDetailDto(detail(null), IST);
    expect(punctuality.arrivalAt).toBe("09:49");
    expect(punctuality.notObservedReason).toBeNull();
  });

  test("multiple adults refuses even when the timetable is complete", () => {
    // The two refusals have different causes and must not mask each other: a
    // filled-in timetable is exactly when a wrong number looks most credible.
    const blended = quality({ multiple_adults_detected: true });
    const { punctuality, lesson } = toDetailDto(detail(blended), IST);
    expect(lesson.scheduledStart).toBe("09:50:00");
    expect(punctuality.arrivalMinutesLate).toBeNull();
  });
});

describe("toPunctuality with an attribution report (Phase 3)", () => {
  const blended = (attribution?: unknown) =>
    quality({ multiple_adults_detected: true, attribution } as Partial<DataQuality>);

  test("a HIGH-confidence attribution lifts the refusal", () => {
    const p = toDetailDto(detail(blended({ confidence: "high", reason: "x" })), IST).punctuality;
    expect(p.arrivalAt).not.toBeNull();
    expect(p.notObservedReason).toBeNull();
  });

  test("a MEDIUM-confidence attribution keeps the numbers withheld, with ITS reason", () => {
    const reason = "1 other adult left while this one remained; 24s is too little to grade on.";
    const p = toDetailDto(detail(blended({ confidence: "medium", reason })), IST).punctuality;
    expect(p.arrivalAt).toBeNull();
    expect(p.notObservedReason).toBe(reason);
  });

  test("no attribution report at all falls back to the Phase 0 refusal", () => {
    const p = toDetailDto(detail(blended()), IST).punctuality;
    expect(p.arrivalAt).toBeNull();
    expect(p.notObservedReason).toContain("adult");
  });
});

describe("previousTeacher: the other lesson in this file", () => {
  // The real handover, as attribution reported it: the period-3 teacher from
  // 246.9 s; the period-2 teacher present at t=0, gone at 335.8 s (09:55:34).
  const handover = (over: Partial<DataQuality["attribution"] & object> = {}) =>
    quality({
      multiple_adults_detected: true,
      attribution: {
        confidence: "medium",
        reason: "1 other adult left while this one remained.",
        chosen_track_no: 1,
        period_known: true,
        splits: 2,
        candidates: [
          {
            track_no: 1,
            first_ms: 246_929,
            last_ms: 359_880,
            present_ms: 113_000,
            in_period_ms: 113_000,
            handed_over: false,
            segments: 3,
          },
          {
            track_no: 2,
            first_ms: 0,
            last_ms: 335_800,
            present_ms: 328_000,
            in_period_ms: 326_000,
            handed_over: true,
            segments: 3,
          },
        ],
        ...over,
      },
    } as Partial<DataQuality>);

  test("her departure is observed, measured into THIS period", () => {
    const p = toDetailDto(detail(handover()), IST).previousTeacher;
    expect(p.state).toBe("observed");
    expect(p.departureAt).toBe("09:55");
    expect(p.periodStart).toBe("09:50");
    // 335.8 s after a bell at +2 s
    expect(p.departureMinutesIntoPeriod).toBe(5.6);
    // Her own bell (09:25 at this school — a 25-minute break precedes period
    // 3) is neither covered by this file nor known without the timetable table.
    expect(p.previousPeriodEndKnown).toBe(false);
    expect(p.adultsAtBell).toBe(1);
    expect(p.reason).toContain("left at 09:55");
  });

  test("an adult who left before this period began is reported as such", () => {
    // Recording from 09:45, this period's bell at 09:50: she leaves at 09:46:40.
    const d = detail(
      handover({
        candidates: [
          {
            track_no: 1,
            first_ms: 400_000,
            last_ms: 2_700_000,
            present_ms: 1,
            in_period_ms: 1,
            handed_over: false,
            segments: 1,
          },
          {
            track_no: 2,
            first_ms: 0,
            last_ms: 100_000,
            present_ms: 1,
            in_period_ms: 1,
            handed_over: true,
            segments: 1,
          },
        ],
      }),
    );
    // 09:45 IST
    d.video.recordingStartedAt = new Date("2026-08-17T04:15:00.000Z");
    const p = toDetailDto(d, IST).previousTeacher;
    expect(p.state).toBe("observed");
    expect(p.departureMinutesIntoPeriod).toBe(-3.3);
    expect(p.departureAt).toBe("09:46");
  });

  test("with nobody else at the bell there is no previous teacher to report", () => {
    const p = toDetailDto(
      detail(
        handover({
          candidates: [
            {
              track_no: 1,
              first_ms: 0,
              last_ms: 359_880,
              present_ms: 1,
              in_period_ms: 1,
              handed_over: false,
              segments: 1,
            },
          ],
        }),
      ),
      IST,
    ).previousTeacher;
    expect(p.state).toBe("none");
    expect(p.departureAt).toBeNull();
  });

  test("a recording that starts after the bell cannot say who was there at it", () => {
    const d = detail(handover());
    // 09:55 IST, bell 09:50
    d.video.recordingStartedAt = new Date("2026-08-17T04:25:00.000Z");
    const p = toDetailDto(d, IST).previousTeacher;
    expect(p.state).toBe("not_observed");
    expect(p.reason).toContain("starts after the 09:50 bell");
  });

  test("an undetermined attribution withholds it with the attribution's reason", () => {
    const p = toDetailDto(
      detail(handover({ confidence: "low", reason: "too close to call" })),
      IST,
    ).previousTeacher;
    expect(p.state).toBe("withheld");
    expect(p.reason).toBe("too close to call");
  });

  test("two handed-over adults at the bell: the last to leave, and say so", () => {
    const p = toDetailDto(
      detail(
        handover({
          candidates: [
            {
              track_no: 1,
              first_ms: 246_929,
              last_ms: 359_880,
              present_ms: 1,
              in_period_ms: 1,
              handed_over: false,
              segments: 3,
            },
            {
              track_no: 2,
              first_ms: 0,
              last_ms: 33_000,
              present_ms: 1,
              in_period_ms: 1,
              handed_over: true,
              segments: 1,
              // the head-only lane
            },
            {
              track_no: 3,
              first_ms: 7_800,
              last_ms: 335_800,
              present_ms: 1,
              in_period_ms: 1,
              handed_over: true,
              segments: 3,
            },
          ],
        }),
      ),
      IST,
    ).previousTeacher;
    expect(p.state).toBe("observed");
    expect(p.departureAt).toBe("09:55");
    expect(p.adultsAtBell).toBe(2);
    expect(p.reason).toContain("2 such adults");
  });

  test("the single-teacher lesson is untouched: no previous teacher, nothing withheld", () => {
    const p = toDetailDto(detail(quality()), IST).previousTeacher;
    // no attribution report on a pre-Phase-3 row
    expect(p.state).toBe("not_observed");
    const p2 = toDetailDto(
      detail(
        handover({
          candidates: [
            {
              track_no: 1,
              first_ms: 0,
              last_ms: 2_700_000,
              present_ms: 1,
              in_period_ms: 1,
              handed_over: false,
              segments: 1,
            },
          ],
          confidence: "high",
        }),
      ),
      IST,
    ).previousTeacher;
    expect(p2.state).toBe("none");
  });
});
