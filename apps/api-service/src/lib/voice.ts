import { RAISED_DB, raisedVoiceEvents, type RaisedVoiceEvent } from "@api/lib/loudness";
import { languageMix, type Language } from "@api/lib/segment";

/**
 * Group D of docs/teacher-measurements.md, as arithmetic over stored
 * sentences — the same shape as R1-R6 over presence intervals: nothing here
 * is stored, so a changed definition is a re-read, not a re-transcription.
 *
 * Two facts decide everything below and neither comes from the transcript
 * alone:
 *
 *  - WHICH SPEAKER IS THE TEACHER. The diarizer labels voices A, B, C; the
 *    video says when the attributed teacher was in the room. The voice that
 *    carries most of the speech while she is present is hers. On the real
 *    handover recording that is what separates the period-2 teacher's opening
 *    minutes (speaker A) from the period-3 teacher (speaker B) — presence
 *    alone would not, since both were in the room at the bell.
 *  - WHAT COUNTS AS A QUESTION. "ठीक है?", "समझे?" and "right?" are
 *    punctuation, not questions; counting question marks overstates R20
 *    several times over. Until the labelling pass exists the split here is a
 *    word list, and the number is reported as PROVISIONAL, not as a fact.
 */

export interface VoiceUtterance {
  idx: number;
  speaker: string;
  startMs: number;
  endMs: number;
  text: string;
  confidence: number | null;
  language: string | null;
  /** Mean RMS of the sentence's audio in dBFS, from the loudness pass; null until it runs. */
  rmsDb?: number | null;
}

export type Interval = [number, number];

export interface VoiceInput {
  utterances: VoiceUtterance[];
  durationMs: number | null;
  /** The attributed teacher's presence intervals, or null when the video
   *  half has nothing trustworthy to say (blended, not analysed). */
  presenceIntervals: Interval[] | null;
  audioStatus: string | null;
  audioError: string | null;
}

export type Tier = "high" | "medium" | "low";

export interface TeacherVoice {
  speaker: string | null;
  confidence: Tier;
  reason: string;
  /** Speech ms per diarized speaker inside her presence, for the reader. */
  bySpeaker: { speaker: string; speechMs: number; inPresenceMs: number }[];
}

export interface VoiceReport {
  /** Why nothing below can be read, when that is the case. */
  state: "observed" | "not_observed";
  reason: string | null;
  audioStatus: string | null;
  teacher: TeacherVoice;
  speech: {
    teacherMs: number;
    othersMs: number;
    silenceMs: number;
    teacherShare: number;
    othersShare: number;
    silenceShare: number;
  } | null;
  /** Longest run of the teacher speaking with no pause over MONOLOGUE_GAP_MS
   *  and nobody else speaking. */
  longestStretchMs: number | null;
  wordsPerMinute: number | null;
  /** R20, provisional: from question marks with check-ins filtered by a word
   *  list. `list` is her questions to the class, oldest first, for the reader. */
  questions: {
    state: "provisional";
    toClass: number;
    checkIns: number;
    perTenMinutes: number;
    list: { idx: number; atMs: number; text: string }[];
  } | null;
  /** R21: the languages she used, the share of her speech in each (a
   *  code-switched sentence splits by its words), and switches per minute. */
  languages: {
    shares: { language: Language; speechMs: number; share: number }[];
    count: number;
    switchesPerMinute: number;
  } | null;
  /** R17: her sentences RAISED_DB above her own median, sustained; merged into episodes. */
  raisedVoice: {
    state: "observed" | "not_observed";
    reason: string | null;
    events: RaisedVoiceEvent[];
    count: number;
    perTenMinutes: number | null;
    baselineDb: number | null;
    thresholdDb: number;
  } | null;
  /** R22 for the microphone. */
  coverage: {
    transcribedMs: number;
    transcribedShare: number;
    meanConfidence: number | null;
    sentences: number;
    words: number;
  } | null;
  /** What rests on phrase patterns until the labelling pass exists, spelled
   *  out so a provisional number is never read as a fact. */
  pendingLabels: string[];
}

/** A language counts as used at this share of her speech. */
export const LANGUAGE_MIN_SHARE = 0.05;
/** Sentences of one speaker closer than this are one stretch of speech. */
export const MONOLOGUE_GAP_MS = 2_000;
/** Below this share of in-presence speech, the dominant voice is a guess. */
export const TEACHER_SHARE_HIGH = 0.6;
export const TEACHER_SHARE_MEDIUM = 0.45;
/** ...and must lead the next voice by this much. */
export const TEACHER_LEAD_HIGH = 2.0;
export const TEACHER_LEAD_MEDIUM = 1.3;

// Check-ins: a question mark on one of these is a verbal full stop. Latin
// entries cover the romanised forms the transcript actually produces.
const CHECK_INS = new Set(
  [
    "ok",
    "okay",
    "right",
    "yes",
    "no",
    "clear",
    "understood",
    "correct",
    "got it",
    "done",
    "fine",
    "alright",
    "yeah",
    "hmm",
    "na",
    "haan",
    "han",
    "hai na",
    "theek hai",
    "thik hai",
    "samjhe",
    "samajh gaye",
    "samjha",
    "samajh aaya",
    "ठीक",
    "ठीक है",
    "ठीक है ना",
    "समझे",
    "समझ गए",
    "समझ आया",
    "समझा",
    "हाँ",
    "हां",
    "ना",
    "है ना",
    "हैं ना",
    "क्लियर",
    "ओके",
  ].map(normalise),
);

function normalise(text: string): string {
  return text
    .toLowerCase()
    .replace(/[.?!।॥,;:"'“”‘’()\-–—]/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

/**
 * A question mark that is a verbal full stop: the whole sentence is a
 * check-in ("ठीक है?"), it is two words or fewer, or it is a statement with a
 * check-in tagged on after the last comma ("...page open करो, ओके?").
 */
export function isCheckIn(text: string): boolean {
  const norm = normalise(text);
  if (CHECK_INS.has(norm) || countWords(norm) <= 2) return true;
  const comma = text.lastIndexOf(",");
  if (comma >= 0) {
    const tail = normalise(text.slice(comma + 1));
    if (tail && (CHECK_INS.has(tail) || countWords(tail) === 1)) return true;
  }
  return false;
}

function overlapMs(a: Interval, b: Interval): number {
  return Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
}

function unionMs(intervals: Interval[]): number {
  const sorted = [...intervals].sort((x, y) => x[0] - y[0]);
  let total = 0;
  let cur: Interval | null = null;
  for (const iv of sorted) {
    if (cur && iv[0] <= cur[1]) {
      cur = [cur[0], Math.max(cur[1], iv[1])];
    } else {
      if (cur) total += cur[1] - cur[0];
      cur = [iv[0], iv[1]];
    }
  }
  if (cur) total += cur[1] - cur[0];
  return total;
}

function countWords(text: string): number {
  return text.split(/\s+/u).filter(Boolean).length;
}

export function teacherSpeaker(
  utterances: VoiceUtterance[],
  presence: Interval[] | null,
): TeacherVoice {
  const speakers = new Map<string, { speechMs: number; inPresenceMs: number }>();
  for (const u of utterances) {
    const s = speakers.get(u.speaker) ?? { speechMs: 0, inPresenceMs: 0 };
    const len = Math.max(0, u.endMs - u.startMs);
    s.speechMs += len;
    if (presence) {
      for (const iv of presence) s.inPresenceMs += overlapMs([u.startMs, u.endMs], iv);
    }
    speakers.set(u.speaker, s);
  }
  const bySpeaker = [...speakers.entries()]
    .map(([speaker, s]) => ({ speaker, ...s }))
    .sort((a, b) => b.inPresenceMs - a.inPresenceMs || b.speechMs - a.speechMs);
  if (bySpeaker.length === 0) {
    return { speaker: null, confidence: "low", reason: "No speech was transcribed.", bySpeaker };
  }
  const top = bySpeaker[0];
  if (!top) {
    return { speaker: null, confidence: "low", reason: "No speech was transcribed.", bySpeaker };
  }

  if (!presence || presence.length === 0) {
    // No trustworthy presence: the loudest voice is probably hers, and that
    // "probably" is the whole confidence.
    const total = bySpeaker.reduce((s, x) => s + x.speechMs, 0);
    const share = total > 0 ? top.speechMs / total : 0;
    return {
      speaker: top.speaker,
      confidence: "low",
      reason:
        `Speaker ${top.speaker} carried ${Math.round(share * 100)}% of all speech; without the ` +
        "video's presence timeline that is a guess.",
      bySpeaker,
    };
  }
  const inPresence = bySpeaker.reduce((s, x) => s + x.inPresenceMs, 0);
  if (inPresence === 0) {
    return {
      speaker: null,
      confidence: "low",
      reason: "Nothing was said while the teacher was in the room.",
      bySpeaker,
    };
  }
  const share = top.inPresenceMs / inPresence;
  // A share alone is not a call: two voices at 50% each is nobody. The top
  // voice must also lead the next one clearly.
  const second = bySpeaker[1]?.inPresenceMs ?? 0;
  const lead = top.inPresenceMs / Math.max(second, 1);
  const confidence: Tier =
    share >= TEACHER_SHARE_HIGH && lead >= TEACHER_LEAD_HIGH
      ? "high"
      : share >= TEACHER_SHARE_MEDIUM && lead >= TEACHER_LEAD_MEDIUM
        ? "medium"
        : "low";
  return {
    speaker: confidence === "low" ? null : top.speaker,
    confidence,
    reason:
      `Speaker ${top.speaker} carried ${Math.round(share * 100)}% of the speech while the ` +
      `teacher was in the room` +
      (confidence === "low" ? " — too little to call that voice hers." : "."),
    bySpeaker,
  };
}

function notObserved(input: VoiceInput, teacher: TeacherVoice, reason: string): VoiceReport {
  return {
    state: "not_observed",
    reason,
    audioStatus: input.audioStatus,
    teacher,
    speech: null,
    longestStretchMs: null,
    wordsPerMinute: null,
    questions: null,
    languages: null,
    raisedVoice: null,
    coverage: null,
    pendingLabels: PENDING_LABELS,
  };
}

const PENDING_LABELS = [
  "R7-R12 lesson start and end",
  "R13-R16 closure, continuation, homework, pack-up",
  "R18 attention requests",
  "R19 off-lesson drift",
  "R20 questions",
];

export function voiceReport(input: VoiceInput): VoiceReport {
  const utts = [...input.utterances].sort((a, b) => a.startMs - b.startMs || a.idx - b.idx);
  const teacher = teacherSpeaker(utts, input.presenceIntervals);

  if (input.audioStatus !== "done" || utts.length === 0) {
    const why =
      input.audioStatus === "skipped"
        ? (input.audioError ?? "Audio was skipped.")
        : input.audioStatus === "empty"
          ? "Transcription returned no speech."
          : input.audioStatus === "failed"
            ? (input.audioError ?? "Audio analysis failed.")
            : input.audioStatus
              ? `Audio is still ${input.audioStatus}.`
              : "No audio analysis has run for this lesson.";
    return notObserved(input, teacher, why);
  }
  if (!teacher.speaker) {
    return notObserved(input, teacher, teacher.reason);
  }

  const duration = input.durationMs ?? Math.max(...utts.map((u) => u.endMs));
  const hers = utts.filter((u) => u.speaker === teacher.speaker);
  const others = utts.filter((u) => u.speaker !== teacher.speaker);
  const teacherMs = unionMs(hers.map((u) => [u.startMs, u.endMs] as Interval));
  const othersMs = unionMs(others.map((u) => [u.startMs, u.endMs] as Interval));
  const spokenMs = unionMs(utts.map((u) => [u.startMs, u.endMs] as Interval));
  const silenceMs = Math.max(0, duration - spokenMs);

  // Longest stretch: her sentences chained while the gap stays short and
  // nobody else's sentence starts in between.
  let longest = 0;
  let runStart: number | null = null;
  let runEnd = 0;
  for (const u of utts) {
    if (u.speaker === teacher.speaker) {
      if (runStart !== null && u.startMs - runEnd < MONOLOGUE_GAP_MS) {
        runEnd = Math.max(runEnd, u.endMs);
      } else {
        runStart = u.startMs;
        runEnd = u.endMs;
      }
      longest = Math.max(longest, runEnd - runStart);
    } else if (runStart !== null) {
      runStart = null;
    }
  }

  const words = hers.reduce((s, u) => s + countWords(u.text), 0);
  const wordsPerMinute =
    teacherMs > 0 ? Math.round((words / (teacherMs / 60_000)) * 10) / 10 : null;

  let toClass = 0;
  let checkIns = 0;
  const questionList: { idx: number; atMs: number; text: string }[] = [];
  for (const u of hers) {
    if (!/\?["'”’)]*$/u.test(u.text.trim())) continue;
    if (isCheckIn(u.text)) checkIns++;
    else {
      toClass++;
      questionList.push({ idx: u.idx, atMs: u.startMs, text: u.text });
    }
  }

  // A sentence's time is split between the languages by its words, so a
  // Hindi frame carrying English nouns is not booked wholly to either.
  const byLanguage = new Map<Language, number>();
  for (const u of hers) {
    const { hi, en } = languageMix(u.text);
    const total = hi + en;
    if (total === 0) continue;
    const len = Math.max(0, u.endMs - u.startMs);
    if (hi > 0) byLanguage.set("hi", (byLanguage.get("hi") ?? 0) + (len * hi) / total);
    if (en > 0) byLanguage.set("en", (byLanguage.get("en") ?? 0) + (len * en) / total);
  }
  const languageTotal = [...byLanguage.values()].reduce((s, x) => s + x, 0);
  let switches = 0;
  let prevLang: Language | null = null;
  for (const u of hers) {
    const lang = u.language === "hi" || u.language === "en" ? u.language : null;
    if (!lang) continue;
    if (prevLang && lang !== prevLang) switches++;
    prevLang = lang;
  }

  const confs = utts.map((u) => u.confidence).filter((c): c is number => c !== null);

  const measured = hers.filter((u) => u.rmsDb !== null && u.rmsDb !== undefined);
  const rv = raisedVoiceEvents(
    hers.map((u) => ({ startMs: u.startMs, endMs: u.endMs, rmsDb: u.rmsDb ?? null })),
  );
  const raisedVoice: VoiceReport["raisedVoice"] =
    measured.length === 0
      ? {
          state: "not_observed",
          reason: "The loudness pass has not run for this lesson; re-run the audio analysis.",
          events: [],
          count: 0,
          perTenMinutes: null,
          baselineDb: null,
          thresholdDb: RAISED_DB,
        }
      : rv.baselineDb === null
        ? {
            state: "not_observed",
            reason: "Too few of her sentences carry a loudness reading to know her baseline.",
            events: [],
            count: 0,
            perTenMinutes: null,
            baselineDb: null,
            thresholdDb: RAISED_DB,
          }
        : {
            state: "observed",
            reason: null,
            events: rv.events,
            count: rv.events.length,
            perTenMinutes:
              duration > 0 ? Math.round((rv.events.length / (duration / 600_000)) * 10) / 10 : null,
            baselineDb: rv.baselineDb,
            thresholdDb: RAISED_DB,
          };

  return {
    state: "observed",
    reason: null,
    audioStatus: input.audioStatus,
    teacher,
    speech: {
      teacherMs,
      othersMs,
      silenceMs,
      teacherShare: duration > 0 ? teacherMs / duration : 0,
      othersShare: duration > 0 ? othersMs / duration : 0,
      silenceShare: duration > 0 ? silenceMs / duration : 0,
    },
    longestStretchMs: longest,
    wordsPerMinute,
    questions: {
      state: "provisional",
      toClass,
      checkIns,
      perTenMinutes: teacherMs > 0 ? Math.round((toClass / (teacherMs / 600_000)) * 10) / 10 : 0,
      list: questionList.slice(0, 200),
    },
    languages: {
      shares: [...byLanguage.entries()]
        .map(([language, speechMs]) => ({
          language,
          speechMs: Math.round(speechMs),
          share: languageTotal > 0 ? speechMs / languageTotal : 0,
        }))
        .sort((a, b) => b.speechMs - a.speechMs),
      // A language she used, not one that appeared: at least LANGUAGE_MIN_SHARE of her speech.
      count: [...byLanguage.values()].filter(
        (ms) => languageTotal > 0 && ms / languageTotal >= LANGUAGE_MIN_SHARE,
      ).length,
      switchesPerMinute:
        teacherMs > 0 ? Math.round((switches / (teacherMs / 60_000)) * 10) / 10 : 0,
    },
    raisedVoice,
    coverage: {
      transcribedMs: spokenMs,
      transcribedShare: duration > 0 ? spokenMs / duration : 0,
      meanConfidence:
        confs.length > 0
          ? Math.round((confs.reduce((s, c) => s + c, 0) / confs.length) * 1000) / 1000
          : null,
      sentences: utts.length,
      words: utts.reduce((s, u) => s + countWords(u.text), 0),
    },
    pendingLabels: PENDING_LABELS,
  };
}
