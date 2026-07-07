// Derived, purely-arithmetic analytics over the intervals the API already
// sends (presenceIntervals, boardIntervals, occupancy). No new backend data.

export type Interval = [number, number];
export type TeacherState = "absent" | "circulating" | "board";
export type StateSegment = { start: number; end: number; state: TeacherState };

const sortIv = (ivs: Interval[]): Interval[] => ivs.toSorted((a, b) => a[0] - b[0]);

/**
 * Split the lesson into contiguous teacher-state segments:
 * gaps between presence = "absent", presence minus board = "circulating",
 * board (always a subset of presence) = "board".
 */
export function teacherStateSegments(
  presence: Interval[],
  board: Interval[],
  durationMs: number,
): StateSegment[] {
  if (durationMs <= 0) return [];
  const pres = sortIv(presence);
  const brd = sortIv(board);
  const segs: StateSegment[] = [];
  let cursor = 0;
  for (const [ps, pe] of pres) {
    if (ps > cursor) segs.push({ start: cursor, end: ps, state: "absent" });
    let c = Math.max(ps, cursor);
    for (const [bs, be] of brd) {
      const s = Math.max(bs, ps);
      const e = Math.min(be, pe);
      if (e <= s || e <= c) continue;
      if (s > c) segs.push({ start: c, end: s, state: "circulating" });
      segs.push({ start: Math.max(s, c), end: e, state: "board" });
      c = e;
    }
    if (c < pe) segs.push({ start: c, end: pe, state: "circulating" });
    cursor = Math.max(cursor, pe);
  }
  if (cursor < durationMs) segs.push({ start: cursor, end: durationMs, state: "absent" });
  return segs;
}

const sumMs = (ivs: Interval[]): number => ivs.reduce((s, [a, b]) => s + (b - a), 0);

export type LessonStats = {
  presentMs: number;
  boardMs: number;
  circulatingMs: number;
  absentMs: number;
  longestPresentMs: number;
  presenceSegments: number;
  boardShare: number | null; // board / present, null if never present
  board: { count: number; longestMs: number; avgMs: number; firstContactMs: number | null };
};

export function lessonStats(
  presence: Interval[],
  board: Interval[],
  durationMs: number,
): LessonStats {
  const presentMs = sumMs(presence);
  const boardMs = sumMs(board);
  const longestPresentMs = presence.reduce((m, [a, b]) => Math.max(m, b - a), 0);
  const brd = sortIv(board);
  const boardDurations = brd.map(([a, b]) => b - a);
  return {
    presentMs,
    boardMs,
    circulatingMs: Math.max(0, presentMs - boardMs),
    absentMs: Math.max(0, durationMs - presentMs),
    longestPresentMs,
    presenceSegments: presence.length,
    boardShare: presentMs > 0 ? boardMs / presentMs : null,
    board: {
      count: brd.length,
      longestMs: boardDurations.reduce((m, d) => Math.max(m, d), 0),
      avgMs: brd.length ? Math.round(boardMs / brd.length) : 0,
      firstContactMs: brd.length ? brd[0]![0] : null,
    },
  };
}

/**
 * Mean concurrent students over the buckets where anyone was present — undoes
 * the dilution of avg_students by empty pre/post-lesson buckets, so it reads
 * closer to true class size. Additive: does not touch the blessed avg_students.
 */
export function avgStudentsWhileOccupied(occupancy: { students: number }[]): number | null {
  const active = occupancy.filter((b) => b.students > 0);
  if (active.length === 0) return null;
  return active.reduce((s, b) => s + b.students, 0) / active.length;
}

/** First bucket that hits the peak student count (deterministic tie-break). */
export function peakOccupancy(
  occupancy: { ts_ms: number; students: number }[],
): { ts_ms: number; students: number } | null {
  let best: { ts_ms: number; students: number } | null = null;
  for (const b of occupancy) {
    if (!best || b.students > best.students) best = { ts_ms: b.ts_ms, students: b.students };
  }
  return best && best.students > 0 ? best : null;
}
