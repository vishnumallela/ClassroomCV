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

// --------------------------------------------------------------------------- //
// Teacher circulation, from the teacher dwell heatmap. Everything here is
// IMAGE-PLANE coverage, never metric distance in feet, which would need
// camera calibration we do not have. Teacher-only since the 2026-08 KPI
// slimming (the students grid is no longer computed).
// --------------------------------------------------------------------------- //

export type Heatmap = { grid_w: number; grid_h: number; teacher: number[] };

export type Circulation = {
  coverage: number; // fraction of all grid cells her path touched
  focusShare: number; // share of teacher time in her single most-used cell
  spread: number; // normalized dwell entropy 0..1 (Moodoo): 0 = one spot, 1 = even
  style: "presenter" | "supervisor" | "balanced"; // Moodoo-style movement pattern
  samples: number;
};

// Moodoo (Martinez-Maldonado et al.) validates that a teacher's dwell
// distribution separates "presenter/authoritative" (front-anchored, low spread)
// from "supervisor" (mobile, high spread, reaches learners) teaching patterns.
// We compute those from the dwell heatmap; they are image-plane, relative to
// this lesson, and describe MOVEMENT PATTERN only, never teaching quality.
const SPREAD_LOW = 0.35; // below this, dwell is concentrated in a few cells
const SPREAD_HIGH = 0.6; // above this, dwell is well distributed
const FOCUS_HIGH = 0.5; // >half of time in one cell = anchored

export function circulation(hm: Heatmap | null | undefined): Circulation | null {
  if (!hm || hm.grid_w <= 0 || hm.grid_h <= 0) return null;
  const { teacher } = hm;
  const total = teacher.reduce((s, n) => s + n, 0);
  if (total <= 0) return null;

  let teacherCells = 0;
  let topCell = 0;
  let entropy = 0; // -sum p*ln p over occupied teacher cells
  for (const t of teacher) {
    if (t > 0) {
      teacherCells++;
      const p = t / total;
      entropy -= p * Math.log(p);
    }
    if (t > topCell) topCell = t;
  }
  // Normalize entropy by ln(occupied cells) so spread is 0..1 and comparable
  // across rooms regardless of how many cells the teacher visited.
  const spread = teacherCells > 1 ? entropy / Math.log(teacherCells) : 0;
  const focusShare = topCell / total;
  const coverage = teacher.length > 0 ? teacherCells / teacher.length : 0;

  // Classify the movement pattern (Moodoo): anchored + low spread reads as a
  // front-of-room presenter; wide spread across much of the room reads as a
  // supervisor circulating; anything else is balanced.
  let style: Circulation["style"] = "balanced";
  if (focusShare >= FOCUS_HIGH || spread < SPREAD_LOW) {
    style = "presenter";
  } else if (spread >= SPREAD_HIGH && coverage >= 0.15) {
    style = "supervisor";
  }

  return { coverage, focusShare, spread, style, samples: total };
}
