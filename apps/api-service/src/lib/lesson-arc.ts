import { labelSentence, type ClosureType, type SentenceLabels } from "@api/lib/phrases";

/**
 * Groups B and C of docs/teacher-measurements.md — when the lesson started
 * and ended, and how it ended — plus R18 and R19, from the teacher's
 * sentences and the video's board and presence intervals.
 *
 * Every measurement carries a STATE. Nothing here is a fact: the sentences
 * are labelled by phrase tables (lib/phrases.ts) until the labelling pass
 * exists, so what is found is PROVISIONAL and shown with its evidence, and
 * what is not found says so rather than reading as zero. That is R23.
 *
 * Lesson start (R7) is the first sentence that sets a task, corroborated by
 * the first board interaction from the video when one begins near it; with
 * no such sentence, the first board interaction stands in. Lesson end (R9)
 * is the later of the last teaching or task sentence and the last time she
 * left the board, capped at her departure — a class that is still being
 * taught from the board has not ended.
 */

export type Interval = [number, number];
export type State = "observed" | "provisional" | "not_observed";

export interface Evidence {
  idx: number;
  atMs: number;
  text: string;
}

export interface Measure<T> {
  value: T | null;
  state: State;
  reason: string | null;
  evidence: Evidence[];
}

export interface ArcSentence {
  idx: number;
  startMs: number;
  endMs: number;
  text: string;
}

export interface ArcInput {
  /** The teacher's sentences, in time order. */
  sentences: ArcSentence[];
  boardIntervals: Interval[];
  durationMs: number;
  /** Scheduled start and end as offsets into the recording, when known. */
  bellStartMs: number | null;
  bellEndMs: number | null;
  /** Her first and last presence, when the video has them. */
  arrivalMs: number | null;
  departureMs: number | null;
  /** Why there are no sentences, when it is not simply "no transcript". */
  noSentencesReason?: string;
}

export interface LessonArc {
  start: Measure<number> & { corroboratedByBoard: boolean; boardAtMs: number | null };
  startDelayMin: Measure<number>;
  end: Measure<number> & { corroboratedByBoard: boolean; boardLeftMs: number | null };
  durationMin: Measure<number>;
  fitsPeriod: Measure<boolean>;
  /** Positive: minutes past the end bell. Negative: minutes of the period left unused. */
  overrunMin: Measure<number>;
  closure: Measure<ClosureType | "none">;
  continuation: Measure<boolean>;
  homework: Measure<boolean> & { atMs: number | null };
  /** Minutes of the first pack-up instruction against the end bell (negative = before it). */
  packUpMin: Measure<number> & { atMs: number | null };
  attentionRequests: Measure<number> & { perTenMinutes: number | null };
  drift: Measure<{ episodes: number; totalMs: number }>;
}

/** A board interaction this close to the sentence corroborates it. */
export const CORROBORATE_MS = 120_000;
/** The window before the end in which closing talk is looked for. */
export const CLOSURE_WINDOW_MS = 5 * 60_000;
/** Tolerance on either bell before a lesson "does not fit". */
export const FIT_TOLERANCE_MS = 60_000;
/** A pack-up instruction is looked for only this close to the end. */
export const PACK_UP_WINDOW_MS = 15 * 60_000;
/** Consecutive procedural sentences within this gap are one episode. */
export const DRIFT_GAP_MS = 30_000;

const NO_AUDIO = "No transcript for this lesson.";

function measure<T>(
  value: T | null,
  state: State,
  reason: string | null = null,
  evidence: Evidence[] = [],
): Measure<T> {
  return { value, state, reason, evidence };
}

function ev(s: ArcSentence): Evidence {
  return { idx: s.idx, atMs: s.startMs, text: s.text };
}

function minutes(ms: number): number {
  return Math.round((ms / 60_000) * 10) / 10;
}

export function lessonArc(input: ArcInput): LessonArc {
  const NO = input.noSentencesReason ?? NO_AUDIO;
  const labelled = [...input.sentences]
    .sort((a, b) => a.startMs - b.startMs || a.idx - b.idx)
    .map((s) => ({ s, l: labelSentence(s.text) }));
  const hasAudio = labelled.length > 0;
  const firstBoard =
    input.boardIntervals.length > 0 ? Math.min(...input.boardIntervals.map((b) => b[0])) : null;
  const lastBoard =
    input.boardIntervals.length > 0 ? Math.max(...input.boardIntervals.map((b) => b[1])) : null;

  // --- R7 start ------------------------------------------------------------
  const afterArrival = (s: ArcSentence) => input.arrivalMs === null || s.endMs >= input.arrivalMs;
  const firstTask = labelled.find(({ s, l }) => l.setsTask && afterArrival(s));
  let start: LessonArc["start"];
  if (firstTask) {
    const near =
      firstBoard !== null && Math.abs(firstBoard - firstTask.s.startMs) <= CORROBORATE_MS;
    start = {
      ...measure(
        firstTask.s.startMs,
        "provisional",
        near
          ? "First task-setting sentence, with the first board interaction near it."
          : "First task-setting sentence; no board interaction near it.",
        [ev(firstTask.s)],
      ),
      corroboratedByBoard: near,
      boardAtMs: firstBoard,
    };
  } else if (firstBoard !== null) {
    start = {
      ...measure(
        firstBoard,
        "provisional",
        hasAudio
          ? "No task-setting sentence found; the first board interaction stands in."
          : "No transcript; the first board interaction stands in.",
      ),
      corroboratedByBoard: false,
      boardAtMs: firstBoard,
    };
  } else {
    start = {
      ...measure<number>(
        null,
        "not_observed",
        hasAudio ? "No task-setting sentence and no board interaction found." : NO,
      ),
      corroboratedByBoard: false,
      boardAtMs: null,
    };
  }

  // --- R8 start delay -----------------------------------------------------
  const startDelayMin =
    start.value !== null && input.bellStartMs !== null
      ? measure(minutes(start.value - input.bellStartMs), start.state, start.reason, start.evidence)
      : measure<number>(
          null,
          "not_observed",
          start.value === null ? start.reason : "The start bell is not known for this lesson.",
        );

  // --- R9 end ---------------------------------------------------------------
  const teaching = (l: SentenceLabels) =>
    !l.procedure && !l.attentionCue && !l.packUp && !l.homework && !l.continuation;
  const lastTeaching = [...labelled].reverse().find(({ l }) => teaching(l) || l.setsTask);
  const candidates: number[] = [];
  if (lastTeaching) candidates.push(lastTeaching.s.endMs);
  if (lastBoard !== null) candidates.push(lastBoard);
  let end: LessonArc["end"];
  if (candidates.length > 0) {
    let endMs = Math.max(...candidates);
    if (input.departureMs !== null) endMs = Math.min(endMs, input.departureMs);
    const near =
      lastTeaching !== undefined &&
      lastBoard !== null &&
      Math.abs(lastBoard - lastTeaching.s.endMs) <= CORROBORATE_MS * 1.5;
    const leftRoomNear =
      input.departureMs !== null && input.departureMs - endMs <= CORROBORATE_MS * 1.5;
    end = {
      ...measure(
        endMs,
        "provisional",
        lastTeaching
          ? near
            ? "Last teaching sentence, with her leaving the board near it."
            : leftRoomNear
              ? "Last teaching sentence, with her leaving the room near it."
              : "Last teaching sentence or last board interaction, whichever is later."
          : "Last board interaction; no transcript to corroborate it.",
        lastTeaching ? [ev(lastTeaching.s)] : [],
      ),
      corroboratedByBoard: near || leftRoomNear,
      boardLeftMs: lastBoard,
    };
  } else {
    end = {
      ...measure<number>(
        null,
        "not_observed",
        hasAudio ? "No teaching sentence and no board interaction found." : NO,
      ),
      corroboratedByBoard: false,
      boardLeftMs: null,
    };
  }

  const endRef = end.value ?? input.departureMs ?? input.durationMs;

  // --- R10-R12 ------------------------------------------------------------
  const both = start.value !== null && end.value !== null;
  const durationMin = both
    ? measure(minutes(end.value! - start.value!), "provisional")
    : measure<number>(null, "not_observed", start.value === null ? start.reason : end.reason);
  const bells = input.bellStartMs !== null && input.bellEndMs !== null;
  const fitsPeriod =
    both && bells
      ? measure(
          start.value! >= input.bellStartMs! - FIT_TOLERANCE_MS &&
            end.value! <= input.bellEndMs! + FIT_TOLERANCE_MS,
          "provisional",
        )
      : measure<boolean>(
          null,
          "not_observed",
          !bells ? "The bells are not known for this lesson." : durationMin.reason,
        );
  const overrunMin =
    end.value !== null && input.bellEndMs !== null
      ? measure(
          minutes(end.value - input.bellEndMs),
          end.state,
          end.value > input.bellEndMs
            ? "Teaching ran past the end bell."
            : "Minutes of the period left after teaching stopped.",
          end.evidence,
        )
      : measure<number>(
          null,
          "not_observed",
          input.bellEndMs === null ? "The end bell is not known for this lesson." : end.reason,
        );

  // --- R13 closure ---------------------------------------------------------
  const closing = labelled.filter(
    ({ s, l }) =>
      l.closure && s.endMs >= endRef - CLOSURE_WINDOW_MS && s.startMs <= endRef + CORROBORATE_MS,
  );
  const closure: Measure<ClosureType | "none"> = !hasAudio
    ? measure<ClosureType | "none">(null, "not_observed", NO)
    : closing.length > 0
      ? measure<ClosureType | "none">(
          closing[0]!.l.closure,
          "provisional",
          null,
          closing.slice(0, 3).map(({ s }) => ev(s)),
        )
      : measure<ClosureType | "none">(
          "none",
          "provisional",
          "No review, reflection, exit question or summary found in the last five minutes.",
        );

  // --- R14 continuation, R15 homework, R16 pack-up -------------------------
  const cont = labelled.filter(({ l }) => l.continuation);
  const continuation = !hasAudio
    ? measure<boolean>(null, "not_observed", NO)
    : measure(
        cont.length > 0,
        "provisional",
        cont.length > 0 ? null : "No statement that the topic continues next time was found.",
        cont.slice(0, 3).map(({ s }) => ev(s)),
      );
  const hw = labelled.filter(({ l }) => l.homework);
  const homework: LessonArc["homework"] = !hasAudio
    ? { ...measure<boolean>(null, "not_observed", NO), atMs: null }
    : {
        ...measure(
          hw.length > 0,
          "provisional",
          hw.length > 0 ? null : "No homework instruction was found.",
          hw.slice(0, 3).map(({ s }) => ev(s)),
        ),
        atMs: hw[0]?.s.startMs ?? null,
      };
  // A pack-up instruction belongs to the closing minutes; "keep your books
  // away" in the middle of a lesson is housekeeping, and on the real lesson a
  // dictation-era sentence matched. Look only in the last PACK_UP_WINDOW_MS
  // before the end (the departure or the file end when the end is unknown).
  const packFrom = endRef - PACK_UP_WINDOW_MS;
  const pack = labelled.find(({ s, l }) => l.packUp && s.startMs >= packFrom);
  const packUpMin: LessonArc["packUpMin"] = !hasAudio
    ? { ...measure<number>(null, "not_observed", NO), atMs: null }
    : !pack
      ? { ...measure<number>(null, "provisional", "No pack-up instruction was found in the last fifteen minutes."), atMs: null }
      : input.bellEndMs === null
        ? {
            ...measure<number>(null, "not_observed", "The end bell is not known for this lesson.", [
              ev(pack.s),
            ]),
            atMs: pack.s.startMs,
          }
        : {
            ...measure(minutes(pack.s.startMs - input.bellEndMs), "provisional", null, [
              ev(pack.s),
            ]),
            atMs: pack.s.startMs,
          };

  // --- R18 attention requests -----------------------------------------------
  const attention = labelled.filter(({ l }) => l.attentionCue);
  const lessonMs = both ? end.value! - start.value! : input.durationMs;
  const attentionRequests: LessonArc["attentionRequests"] = !hasAudio
    ? { ...measure<number>(null, "not_observed", NO), perTenMinutes: null }
    : {
        ...measure(
          attention.length,
          "provisional",
          null,
          attention.slice(0, 5).map(({ s }) => ev(s)),
        ),
        perTenMinutes:
          lessonMs > 0 ? Math.round((attention.length / (lessonMs / 600_000)) * 10) / 10 : null,
      };

  // --- R19 drift: runs of procedural talk ----------------------------------
  let episodes = 0;
  let totalMs = 0;
  let run: { start: number; end: number; n: number } | null = null;
  const driftEvidence: Evidence[] = [];
  const closeRun = () => {
    if (run && run.n >= 2) {
      episodes++;
      totalMs += run.end - run.start;
    }
    run = null;
  };
  for (const { s, l } of labelled) {
    if (l.procedure && !l.setsTask) {
      if (run && s.startMs - run.end <= DRIFT_GAP_MS) {
        run.end = Math.max(run.end, s.endMs);
        run.n++;
        if (run.n === 2 && driftEvidence.length < 3) driftEvidence.push(ev(s));
      } else {
        closeRun();
        run = { start: s.startMs, end: s.endMs, n: 1 };
      }
    } else {
      closeRun();
    }
  }
  closeRun();
  const drift = !hasAudio
    ? measure<{ episodes: number; totalMs: number }>(null, "not_observed", NO)
    : measure(
        { episodes, totalMs },
        "provisional",
        "Runs of administrative talk (notebooks, planners, signatures, fees) stand in for off-lesson drift until the labelling pass.",
        driftEvidence,
      );

  return {
    start,
    startDelayMin,
    end,
    durationMin,
    fitsPeriod,
    overrunMin,
    closure,
    continuation,
    homework,
    packUpMin,
    attentionRequests,
    drift,
  };
}
