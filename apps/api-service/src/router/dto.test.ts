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
