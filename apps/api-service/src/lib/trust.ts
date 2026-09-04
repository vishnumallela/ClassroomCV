import type { LessonArc, State } from "@api/lib/lesson-arc";
import type { VoiceReport } from "@api/lib/voice";

/**
 * Group E — R22 (how much the system could see and hear) and R23 (it says
 * Not Observed rather than guessing) — as one list over every measurement in
 * docs/teacher-measurements.md, so a reader sees at a glance which numbers on
 * the page are facts, which are provisional, and which were withheld and why.
 */

export interface TrustItem {
  id: string;
  name: string;
  state: State;
  reason: string | null;
}

export interface TrustSummary {
  observed: number;
  provisional: number;
  notObserved: number;
  items: TrustItem[];
}

interface PunctualityLike {
  arrivalAt: string | null;
  departureAt: string | null;
  arrivalMinutesLate: number | null;
  departureMinutesLate: number | null;
  presenceShareOfPeriod: number | null;
  notObservedReason: string | null;
}

export function trustItems(input: {
  punctuality: PunctualityLike;
  videoAnalysed: boolean;
  videoQuality: string | null;
  videoCoverage: number | null;
  arc: LessonArc;
  voice: VoiceReport;
}): TrustSummary {
  const { punctuality: p, arc, voice } = input;
  const noVideo = p.notObservedReason ?? "The video analysis has not produced a teacher timeline.";
  const noBells = "The bells are not known for this lesson.";
  const obs = (ok: boolean, reason: string): [State, string | null] =>
    ok ? ["observed", null] : ["not_observed", reason];
  const m = (x: { state: State; reason: string | null }): [State, string | null] => [
    x.state,
    x.reason,
  ];

  const rows: [string, string, [State, string | null]][] = [
    ["R1", "Arrival time", obs(p.arrivalAt !== null, noVideo)],
    [
      "R2",
      "Arrival against the bell",
      obs(p.arrivalMinutesLate !== null, p.arrivalAt === null ? noVideo : noBells),
    ],
    ["R3", "Departure time", obs(p.departureAt !== null, noVideo)],
    [
      "R4",
      "Departure against the bell",
      obs(p.departureMinutesLate !== null, p.departureAt === null ? noVideo : noBells),
    ],
    [
      "R5",
      "Time in the room",
      obs(p.presenceShareOfPeriod !== null, p.arrivalAt === null ? noVideo : noBells),
    ],
    ["R6", "Mid-lesson absences", obs(input.videoAnalysed, noVideo)],
    ["R7", "Lesson start", m(arc.start)],
    ["R8", "Start delay", m(arc.startDelayMin)],
    ["R9", "Lesson end", m(arc.end)],
    ["R10", "Lesson duration", m(arc.durationMin)],
    ["R11", "Did the lesson fit the period", m(arc.fitsPeriod)],
    ["R12", "Overrun or underrun", m(arc.overrunMin)],
    ["R13", "Closure, and its type", m(arc.closure)],
    ["R14", "Continuation", m(arc.continuation)],
    ["R15", "Homework set", m(arc.homework)],
    ["R16", "Pack-up instruction", m(arc.packUpMin)],
    [
      "R17",
      "Raised-voice events",
      voice.raisedVoice
        ? [voice.raisedVoice.state, voice.raisedVoice.reason]
        : ["not_observed", voice.reason],
    ],
    ["R18", "Attention requests", m(arc.attentionRequests)],
    ["R19", "Off-lesson drift", m(arc.drift)],
    [
      "R20",
      "Questions asked",
      voice.questions
        ? ["provisional", "Question marks with check-ins set aside, until the labelling pass."]
        : ["not_observed", voice.reason],
    ],
    [
      "R21",
      "Languages used",
      voice.languages
        ? ["observed", "Read from each sentence's script."]
        : ["not_observed", voice.reason],
    ],
    [
      "R22",
      "Observation coverage",
      input.videoAnalysed || voice.coverage
        ? [
            "observed",
            `Video: ${input.videoAnalysed ? `${Math.round((input.videoCoverage ?? 0) * 100)}% coverage, ${input.videoQuality ?? "unrated"}` : "not analysed"}. Audio: ${voice.coverage ? `${Math.round(voice.coverage.transcribedShare * 100)}% transcribed` : "none"}.`,
          ]
        : ["not_observed", "Neither sensor has produced a timeline."],
    ],
  ];
  const items: TrustItem[] = rows.map(([id, name, [state, reason]]) => ({
    id,
    name,
    state,
    reason,
  }));
  const notObserved = items.filter((i) => i.state === "not_observed").length;
  items.push({
    id: "R23",
    name: "Not Observed",
    state: "observed",
    reason: `${notObserved} of 22 measurements withheld rather than guessed.`,
  });
  return {
    observed: items.filter((i) => i.state === "observed").length,
    provisional: items.filter((i) => i.state === "provisional").length,
    notObserved,
    items,
  };
}
