import { describe, expect, test } from "bun:test";
import { isCheckIn, teacherSpeaker, voiceReport, type VoiceUtterance } from "@api/lib/voice";

let n = 0;
function u(
  speaker: string,
  startMs: number,
  endMs: number,
  text: string,
  language: string | null = "en",
): VoiceUtterance {
  return { idx: n++, speaker, startMs, endMs, text, confidence: 0.9, language };
}

// The real handover's shape: speaker A talks for the first five minutes
// (the period-2 teacher), speaker B thereafter; the attributed teacher is
// present from 247 s to the end.
const PRESENCE: [number, number][] = [[247_000, 2_700_000]];
const HANDOVER = [
  u("A", 0, 60_000, "Lunch break is over, get back to your seats.", "en"),
  u("A", 70_000, 240_000, "Who is roll number twelve? Give the notebook here.", "en"),
  u("B", 300_000, 600_000, "अब हम वर्कशीट देखेंगे।", "hi"),
  u("A", 610_000, 615_000, "Yes ma'am.", "en"),
  u("B", 620_000, 1_200_000, "Page number twenty. Who can read the first line?", "en"),
  u("B", 1_201_000, 1_500_000, "ठीक है?", "hi"),
  u("B", 1_500_000, 1_800_000, "अब अगला प्रश्न।", "hi"),
  u("B", 1_900_000, 2_500_000, "What is the past tense of go?", "en"),
  u("B", 2_500_000, 2_600_000, "Right?", "en"),
];

describe("teacherSpeaker", () => {
  test("the voice that carries the speech while she is present is hers", () => {
    const t = teacherSpeaker(HANDOVER, PRESENCE);
    expect(t.speaker).toBe("B");
    expect(t.confidence).toBe("high");
    expect(t.reason).toContain("while the teacher was in the room");
  });

  test("without a presence timeline the dominant voice is only a guess", () => {
    const t = teacherSpeaker(HANDOVER, null);
    expect(t.speaker).toBe("B");
    expect(t.confidence).toBe("low");
  });

  test("two voices sharing her presence evenly is not a call", () => {
    const even = [u("A", 300_000, 900_000, "one", "en"), u("B", 900_000, 1_500_000, "two", "en")];
    const t = teacherSpeaker(even, PRESENCE);
    expect(t.speaker).toBeNull();
    expect(t.confidence).toBe("low");
  });

  test("no speech, no teacher", () => {
    expect(teacherSpeaker([], PRESENCE).speaker).toBeNull();
  });
});

describe("isCheckIn", () => {
  test("a tagged-on check-in is punctuation, a real question is not", () => {
    expect(isCheckIn("ठीक है?")).toBe(true);
    expect(isCheckIn("इन द कंटेंट पेज यू नीड टू डू इट देयर, ओके?")).toBe(true);
    expect(isCheckIn("Please check whether it is done, okay?")).toBe(true);
    expect(isCheckIn("Got it?")).toBe(true);
    expect(isCheckIn("Where is your literacy companion?")).toBe(false);
    expect(isCheckIn("Have you done 18?")).toBe(false);
    expect(isCheckIn("You have done, or the class has done?")).toBe(false);
  });
});

describe("voiceReport", () => {
  const input = {
    utterances: HANDOVER,
    durationMs: 2_700_000,
    presenceIntervals: PRESENCE,
    audioStatus: "done",
    audioError: null,
  };

  test("speech is split three ways and sums to the recording", () => {
    const r = voiceReport(input);
    expect(r.state).toBe("observed");
    expect(r.speech).not.toBeNull();
    const s = r.speech!;
    expect(s.teacherMs).toBe(300_000 + 580_000 + 599_000 + 700_000);
    expect(s.othersMs).toBe(60_000 + 170_000 + 5_000);
    expect(s.teacherMs + s.othersMs + s.silenceMs).toBe(2_700_000);
    expect(s.teacherShare).toBeCloseTo(s.teacherMs / 2_700_000);
  });

  test("the longest stretch chains her sentences across short gaps only", () => {
    const r = voiceReport(input);
    // 620 s -> 1800 s: two sentences 1 s apart; the 100 s gap before 1900 s breaks it
    expect(r.longestStretchMs).toBe(1_180_000);
  });

  test("questions are counted from her sentences with check-ins set aside, and marked provisional", () => {
    const r = voiceReport(input);
    expect(r.questions).toMatchObject({ state: "provisional", toClass: 2, checkIns: 2 });
  });

  test("languages: the set, the share of each, and the switch rate", () => {
    const r = voiceReport(input);
    expect(r.languages?.count).toBe(2);
    const en = r.languages?.shares.find((s) => s.language === "en");
    const hi = r.languages?.shares.find((s) => s.language === "hi");
    expect((en?.share ?? 0) + (hi?.share ?? 0)).toBeCloseTo(1);
    // hi, en, hi, hi, en, en across her sentences: three switches
    expect(r.languages?.switchesPerMinute).toBe(
      Math.round((3 / (r.speech!.teacherMs / 60_000)) * 10) / 10,
    );
  });

  test("audio that was skipped reports Not Observed with the reason, not zeros", () => {
    const r = voiceReport({
      ...input,
      audioStatus: "skipped",
      audioError: "This recording has no audio track.",
    });
    expect(r.state).toBe("not_observed");
    expect(r.reason).toBe("This recording has no audio track.");
    expect(r.speech).toBeNull();
    expect(r.pendingLabels.length).toBeGreaterThan(0);
  });

  test("a lesson still transcribing says so", () => {
    const r = voiceReport({ ...input, utterances: [], audioStatus: "transcribing" });
    expect(r.state).toBe("not_observed");
    expect(r.reason).toContain("transcribing");
  });
});
