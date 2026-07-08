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

// --------------------------------------------------------------------------- //
// Lesson rhythm: punctuality, settle time, unsupervised stretches. All derived
// from the presence + occupancy series the API already sends; nothing implies
// metric distance, named attendance, or engagement (see docs decision doc).
// --------------------------------------------------------------------------- //

export type LessonRhythm = {
  firstTeacherMs: number | null; // when a teacher-role adult first appears
  settleMs: number | null; // lesson start -> students reach 90% of peak
  longestUnsupervisedMs: number; // longest stretch with no teacher present
  unsupervisedCount: number; // number of such stretches > a tracking-noise floor
};

const UNSUPERVISED_FLOOR_MS = 15_000; // shorter gaps are tracking dropout, not absence

export function lessonRhythm(
  presence: Interval[],
  occupancy: { ts_ms: number; students: number }[],
  durationMs: number,
): LessonRhythm {
  const pres = sortIv(presence);
  const firstTeacherMs = pres.length ? pres[0]![0] : null;

  // Settle: from first student presence until the count first reaches 90% of peak.
  const occ = occupancy.filter((b) => b.students > 0);
  let settleMs: number | null = null;
  if (occ.length > 0) {
    const peak = occ.reduce((m, b) => Math.max(m, b.students), 0);
    const start = occ[0]!.ts_ms;
    const settled = occ.find((b) => b.students >= 0.9 * peak);
    if (settled) settleMs = Math.max(0, settled.ts_ms - start);
  }

  // Unsupervised: gaps between presence intervals, plus the lead/tail if the
  // teacher was absent at the edges, longer than the tracking-noise floor.
  let longest = 0;
  let count = 0;
  const consider = (ms: number) => {
    if (ms > UNSUPERVISED_FLOOR_MS) {
      count++;
      longest = Math.max(longest, ms);
    }
  };
  if (pres.length === 0) {
    longest = durationMs;
    count = durationMs > UNSUPERVISED_FLOOR_MS ? 1 : 0;
  } else {
    consider(pres[0]![0]); // before the teacher first arrives
    for (let i = 1; i < pres.length; i++) consider(pres[i]![0] - pres[i - 1]![1]);
    consider(durationMs - pres[pres.length - 1]![1]); // after the teacher leaves
  }
  return { firstTeacherMs, settleMs, longestUnsupervisedMs: longest, unsupervisedCount: count };
}

// --------------------------------------------------------------------------- //
// Teacher circulation, from the dwell heatmap (teacher vs student per-cell
// sample counts). Everything here is IMAGE-PLANE coverage — never metric
// distance in feet, which would need camera calibration we do not have.
// --------------------------------------------------------------------------- //

export type Heatmap = { grid_w: number; grid_h: number; teacher: number[]; students: number[] };

export type Circulation = {
  coverage: number; // fraction of the active room the teacher's path touched
  amongStudentsShare: number; // share of teacher time in cells where students sit
  focusShare: number; // share of teacher time in her single most-used cell
  reachedBackRows: boolean; // did she reach the row band farthest from the front
  samples: number;
};

export function circulation(hm: Heatmap | null | undefined): Circulation | null {
  if (!hm || hm.grid_w <= 0 || hm.grid_h <= 0) return null;
  const { grid_w, grid_h, teacher, students } = hm;
  const total = teacher.reduce((s, n) => s + n, 0);
  if (total <= 0) return null;

  let teacherCells = 0;
  let activeCells = 0;
  let amongStudents = 0;
  let topCell = 0;
  for (let i = 0; i < teacher.length; i++) {
    const t = teacher[i]!;
    const st = students[i] ?? 0;
    if (t > 0 || st > 0) activeCells++;
    if (t > 0) teacherCells++;
    if (t > 0 && st > 0) amongStudents += t;
    if (t > topCell) topCell = t;
  }

  // "Back rows" = the third of grid rows with the most student mass but the
  // least camera proximity; we approximate it as the student-dense row band
  // and ask whether the teacher ever entered it.
  const rowStudents = Array.from({ length: grid_h }, (_v, r) => {
    let mass = 0;
    for (let c = 0; c < grid_w; c++) mass += students[r * grid_w + c] ?? 0;
    return mass;
  });
  const studentRows = rowStudents
    .map((mass, r) => ({ mass, r }))
    .filter((x) => x.mass > 0)
    .toSorted((a, b) => b.mass - a.mass);
  const backBand = new Set(
    studentRows.slice(0, Math.max(1, Math.round(grid_h / 3))).map((x) => x.r),
  );
  let reachedBackRows = false;
  for (let r = 0; r < grid_h && !reachedBackRows; r++) {
    if (!backBand.has(r)) continue;
    for (let c = 0; c < grid_w; c++) {
      if ((teacher[r * grid_w + c] ?? 0) > 0) {
        reachedBackRows = true;
        break;
      }
    }
  }

  return {
    coverage: activeCells > 0 ? teacherCells / activeCells : 0,
    amongStudentsShare: amongStudents / total,
    focusShare: topCell / total,
    reachedBackRows,
    samples: total,
  };
}
