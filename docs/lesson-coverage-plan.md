# The lesson is the period, not the file — recording protocol and coverage plan

**Status:** Diagnosed 2026-09-03. Phase B′ built 2026-09-04. **Phase D's table, resolver and editor built 2026-09-04** (register view still to do); A, B, C, E not started. School bell times confirmed from the 7B timetable 2026-09-04 (§2 of the attribution plan's 09:50 guess was right; periods are not contiguous).
**Branch:** `feat/rfdetr-pipeline` (not pushed)
**Companion:** `docs/teacher-attribution-plan.md` — that plan fixes *who* the
timeline belongs to. This one fixes *when the timeline is allowed to say
anything*. Phase C below depends on its Phases 2 and 3 — **both landed
2026-09-03** (`e31a413`, `c5e0e20`), so the "until attribution segments exist"
caveats in §6 and Phase C are now lifted; everything else here is independent
of it.

Pick this up cold: the problem, the exact code that makes the assumption, the
recording protocol the school has to follow, and the phases are all below.

---

## 1. The problem in one paragraph

The school records in 45-minute files and the timetable runs in 45-minute
periods, so the pipeline treats **one file as one lesson**. Every punctuality
number (R1–R6) is a subtraction between a video offset and a bell time, read
from a single `videos` row. That holds only when the file starts on the bell
and runs past the end bell. The moment a file boundary falls anywhere else,
the numbers go wrong, and — as with the attribution bug — they go wrong
**silently and in the flattering direction**:

| Situation | What the school will do | What the system reports today |
| --- | --- | --- |
| Teacher arrives after the file that covers the bell has ended | "She was late for period 3 — it's in the next video" | **Nothing** for period 3, or her arrival is credited to period 4 |
| Recording started when she walked in, not on the bell | Someone presses record when the lesson "starts" | **On time**, by construction — lateness is erased before upload |
| Recording stops before the end bell | Camera runs 45 min flat on a period that runs 47 | **Left early**, when it was the camera that left |
| Recording starts a few minutes before the bell with the previous teacher still there | Continuous recording, split at fixed intervals | The attribution bug, `docs/teacher-attribution-plan.md` |

The question that prompted this — *"the video ends at 45 minutes, how do we
show she is late when that is in the next video?"* — is the first row. The
answer is that the lesson has to be defined as the **scheduled window on the
classroom's wall-clock day**, and files are just **coverage** of that window.
A period can be covered by two files; a file can cover parts of two periods.

---

## 2. What the current recording actually looks like

From `docs/teacher-attribution-plan.md` §2, measured 2026-09-03:

| | |
| --- | --- |
| File | `~/Downloads/20082026/fixed/4957a8af-3d98-4c22-80ef-9d268b51afe3.mp4` |
| Duration | 2700.9 s — exactly 45:00 |
| Burned-in clock at t=0 | `2026-08-17 09:49:58` |
| Wall-clock range | 09:49:58 → 10:34:59 |
| `creation_time` | **absent** — stripped by ffmpeg (`encoder: Lavf62.x`) |
| Teacher's real arrival | 09:54:25 (offset 267 s) |

So this file *does* contain her arrival, because it happened to start two
seconds before the bell. Her 4.5 minutes of lateness is in **this** video. The
"next video" problem arises only when the file boundary falls between the bell
and her arrival — for example a file running 09:10 → 09:55 on a 09:50 bell
with an arrival at 09:57. That is not a rare edge case; it is what a
continuously-recording camera split at fixed intervals produces most days.

The 45:00 duration and the two-second lead before the bell suggest the camera
is **already on a fixed schedule aligned to the timetable**. If that is true,
row 1 of the table above mostly does not happen, and rows 2–3 are the live
risks. **Confirm with the school before building Phase C** — see §8.

---

## 3. Where the assumption lives

| File / line | What it does | Why it is wrong |
| --- | --- | --- |
| `apps/api-service/src/db/schema/index.ts:120-133` | `recordingStartedAt`, `lessonDate`, `period`, `scheduledStart`, `scheduledEnd`, `hasFollowingPeriod` are columns **on `videos`** | the timetable is a fact about a classroom's day, stored as a fact about a file |
| `apps/api-service/src/router/dto.ts` `toPunctuality` | `const first = presence[0]; const last = presence.at(-1)` | first/last interval **of one file**, whatever the file covered |
| `apps/api-service/src/lib/school-time.ts` `minutesAgainstSchedule` | `actual − scheduled`, rounded | never asks whether the recording **reached** the scheduled instant; an end-of-file departure is reported as leaving early |
| `apps/api-service/src/db/queries.ts:401` `getVideoDetail` | loads one video's analytics | no notion of sibling files on the same classroom-day |
| `apps/api-service/src/lib/media.ts:87` | anchor = container `creation_time` only | any re-encode strips it; this file has none |
| `services/ml-service/app/teacher.py` `_co_presence` | counts two adults **within one file** | a handover split across two files is never co-present in either, so Phase 0's refusal does not fire (§6, trap 1) |
| `apps/frontend/src/components/lesson-details-card.tsx:186-211` | renders `arrivalMinutesLate` or "Not Observed" | two states; the honest answer needs three (§5) |

The ML service is otherwise **not implicated**. `presence_intervals` are file
offsets and should stay that way; anchoring to the wall clock is the API's job
and already happens at read time.

---

## 4. The recording protocol the school must follow

This is the part that is not code. No amount of stitching recovers lateness
that was never recorded. Hand the school this list; it is short on purpose.

1. **Record on the bell, not on the teacher.** The camera or NVR starts on a
   fixed schedule. Nobody in the room presses record. A recording that begins
   when she walks in has already destroyed the number it exists to measure.
2. **Run past the end bell.** Either record continuously through the school day
   per classroom, or start each file at the bell and stop it **at least 5
   minutes after the next bell**. Adjacent files may overlap; they must not
   leave a gap around a bell.
3. **Upload the camera's original files.** Do not trim, re-encode or "fix" them
   first. The re-encode is what stripped `creation_time` from this recording.
   If the school's process requires ffmpeg, it needs `-map_metadata 0`, and
   Phase E's clock OCR is the backstop for when that is forgotten.
4. **Upload every file for the classroom-day, including ones with nobody in
   them.** An empty file is evidence: it proves the room had no teacher between
   two clock times. Skipping it turns "absent" into "Not Observed".
5. **Fill in the timetable once, not per file.** Bell times are a property of
   the classroom's week, not of a video. Since Phase D (2026-09-04) they are
   typed once on the classroom's Settings page and every lesson in the room
   resolves its bells from there; typing them on a video is now an override.

Folder and file names are **not** anchors. The folder this recording came in is
named `20082026`; the lesson is 17 August.

---

## 5. Target semantics — three states, not two

Every Group A number gets a **state**, and the state is derived from
**coverage**: the union of the anchored wall-clock ranges of every file on the
same classroom-day that overlaps the scheduled window.

```
coverage = ⋃ [recording_started_at, recording_started_at + duration_ms]
           over files with the same classroom and lesson date
```

### Arrival (R1, R2)

| State | Condition | Reported as |
| --- | --- | --- |
| **observed** | first presence interval starts inside coverage, and the room was **empty at P1** or the file is a clean single-adult file | "arrived 09:54, 4.5 min late" |
| **lower bound** | coverage includes P1, no presence from P1 to the end of coverage | "not seen by 10:35 when the recording ended — **at least 45 min late**" |
| **not observed** | coverage does not include P1 | "no recording covers the 09:50 bell" |

The lower bound tightens to an exact value the moment the next file is
uploaded, with no re-analysis, because all of this is derived at read time.

### Departure (R3, R4)

| State | Condition | Reported as |
| --- | --- | --- |
| **observed** | last presence ends inside coverage **and coverage extends past P2** (or past her exit by a margin) | "left 10:33, 2 min early" |
| **lower bound** | she is present at the end of coverage, coverage ends before P2 | "still in the room at 10:34 when the recording ended" — **never** "left early" |
| **not observed** | coverage does not include P2 and she was not present at its end | "no recording covers the 10:35 bell" |

### Time in the room (R5)

Denominator stays the scheduled period, as today. Add the **covered fraction of
the period** alongside it, so "44% present" and "44% present, 100% covered"
stop reading the same as "44% present, 50% covered".

### Mid-lesson absences (R6)

Unchanged within a file. Across a file boundary, a gap that spans the boundary
is an absence only if coverage is contiguous across it; a gap in coverage is a
gap in coverage, not an absence.

### The DTO shape

`punctuality` gains, per measurement, a state and the coverage facts that
justify it. Additive — nothing existing is renamed:

```ts
arrivalState: "observed" | "lower_bound" | "not_observed";
arrivalMinutesLateAtLeast: number | null;   // set only for lower_bound
departureState: "observed" | "lower_bound" | "not_observed";
coverage: {
  from: string | null;                       // local HH:MM
  to: string | null;
  coversScheduledStart: boolean;
  coversScheduledEnd: boolean;
  fileCount: number;
  coveredShareOfPeriod: number | null;
};
```

`notObservedReason` keeps its current meaning (attribution refused) and is
joined by the coverage reasons above, in words a reader can act on.

---

## 6. Traps found while diagnosing

1. **A handover split across two files bypasses Phase 0.** `_co_presence` runs
   per file. If period 2's teacher leaves at 09:52 in file A (ending 09:52) and
   period 3's teacher enters at 09:52 in file B, neither file ever shows two
   adults, and a stitched timeline reads as one unbroken presence. Phase C must
   not merge presence across a boundary when someone was present at both
   sides of it, until attribution segments exist. See the "empty at P1" rule.
2. **`minutesAgainstSchedule` has no coverage check.** Any file that stops
   before the bell reports an early departure today, on every lesson, not only
   handovers. Phase B fixes this on its own and is the cheapest correct change
   in this document.
3. **`durationMs` is nullable** on `videos`. Coverage needs it; a probe that
   failed to read a duration must produce **not observed**, never a zero-length
   range that makes every lesson look uncovered.
4. **Timetable fields are per video.** Two files covering one period can carry
   two different typed-in bell times. Phase C has to pick one (the file the
   user is looking at) and warn on disagreement; Phase D removes the
   duplication.
5. **`lessonDate` defaults from `recordingStartedAt`** (`lessonClock`). A file
   that starts at 23:58 the previous evening — a continuous recorder rolling
   over midnight — lands on the wrong day. Not a risk for a school-hours
   schedule; worth a test so it stays that way.
6. **Pydantic silently drops unknown keys** on the ML side
   (`DataQualityOut`). Not touched by this plan, but if any coverage fact ever
   moves into `data_quality`, it needs a field there or it vanishes.

---

## 7. Phases

### Phase A — Recording protocol and upload guardrails (small, do first)

The five rules in §4, as a page the school keeps, plus two guardrails in the
upload path so a bad file is caught at upload rather than at reporting time:

- After probe, show the anchored wall-clock range next to the file. If
  `recordingStartedAt` is null, say **"no start time — punctuality will read
  Not Observed until one is entered"** where the uploader can see it.
- If the timetable fields are filled and the anchored range does **not** cover
  `scheduledStart`, warn at upload: "this file starts 09:55, after the 09:50
  bell".

Touches: `apps/frontend` upload page, `apps/api-service/src/router/videos.ts`
probe response. No schema change. No ML change.

### Phase B — Coverage-aware punctuality, single file (small)

Make `toPunctuality` honest about one file before making it read several.

- Compute `coverage` from `recordingStartedAt` and `durationMs`.
- Departure: if she is present at end of file and the file ends before P2,
  state is **lower bound**, `departureMinutesLate` is null.
- Arrival: if the file covers P1 and there is no presence, state is **lower
  bound** with `arrivalMinutesLateAtLeast = coverage.to − P1`.
- Render the three states in `lesson-details-card.tsx`.

Pure DTO arithmetic; every case is a fixture in `dto.test.ts`. This fixes trap
2 for every lesson already analysed, on the next page load.

### Phase B′ — The other lesson in this file (small) — BUILT 2026-09-04

The question that prompted it: *the period-2 teacher is still in the period-3
recording — when did she leave, and did she finish her lesson?* This is row 4
of §1 seen from the previous period's side, and once attribution exists it is
answerable from **this one file, at read time**:

- Attribution marks her `handed_over`: present at the bell, gone while the
  period's own teacher remained. By the same rule that picks this period's
  teacher (the adult who stays to the end), the adult in the room at the
  **end of the previous period** is that period's teacher.
- So her last sighting here is the previous period's **departure (R3)**, and
  whether she stayed to her own bell is **R4** against it — **but only once her
  bell is known.** The school's timetable (7B, 2026) shows periods are NOT
  back-to-back: period 2 ends 09:25, period 3 starts 09:50, with a 25-minute
  break between. So "this period's start" is not "her period's end", and a
  file that starts on this period's bell does not cover hers. Until Phase D's
  table exists, her departure is reported against **this** period's start
  ("5.6 min into the period") and nothing is claimed about her own bell.
- `dto.ts` `toPreviousTeacher` → `previousTeacher` on the detail DTO, with the
  same three-state honesty as §5: **observed** / **not_observed** (no anchor,
  no timetable, or the file starts after the bell so nobody's presence at it
  is covered) / **withheld** (attribution undetermined) / **none** (nobody
  else was at the bell — a single-teacher lesson renders nothing at all).
  Several handed-over adults at the bell (one person tracked in pieces) →
  the last to leave, and the count is reported so a reader can judge.
- `lesson-details-card.tsx` renders it under "Previous period's teacher".

**Her departure is `left_ms`, not her last sighting (2026-09-04).** On the
full recording a colleague who came in 34 minutes later linked to her by
appearance (two light outfits), so her "last sighting" became 10:32.
Attribution now reports, per adult present at the bell, the end of the
presence run containing the bell (absences under 5 minutes bridged), and
`toPreviousTeacher` uses that. Verified on the full file: *left 09:55 —
5.6 min into the period*, the same answer the trim gave.

On the real handover clip: *left the room 09:55 — 5.6 min into the period.*
That is the period-2 teacher's R3 read off the period-3 file, with no
re-analysis. Her R4 (against 09:25) needs Phase D; it would read **+30 min**,
which with a 25-minute break in between means she stayed in the room through
the break rather than that she taught over — another reason the bell has to
come from the timetable and not be inferred.

**What this file cannot say**, and the card says so: whether she was there for
the *whole* of her period. Her arrival and mid-lesson absences are in the
previous file — Phase C's stitching, which now has attribution's people to
stitch with rather than raw presence.

### Phase C — Coverage across files (medium)

`getVideoDetail` gains the sibling files: same `classroomId`, anchored range
overlapping `[P1 − 30 min, P2 + 30 min]`, status completed. The DTO unions
their coverage and concatenates their `presenceIntervals` on the wall clock.

Merging rule, until attribution segments exist (trap 1):

- **Room empty at P1** across the union → arrival is the first presence after
  P1, in whichever file it lands. Safe, because there is no one to confuse her
  with.
- **Someone present at P1** → if all of that presence is in one file, today's
  single-file behaviour applies (subject to the Phase 0 refusal). If it spans a
  file boundary, **refuse** with the reason "an adult was in the room across a
  recording boundary at the bell; which one this lesson assesses is not yet
  decided". Attribution Phases 2–3 lift this.

Also needed: the classroom-day listing query, and disagreement handling for
trap 4.

### Phase D — Lessons as a first-class thing (medium-large) — TABLE, RESOLVER AND EDITOR BUILT 2026-09-04

The schema comment on `videos.period` already anticipated this: *"A
per-classroom period table can fill this in later without a rewrite."*

Built (`0015_timetable_periods`, `lib/timetable.ts`, `classrooms.setTimetable`,
`components/timetable-card.tsx`):

- `timetable_periods(classroom_id, weekday, slot, label, scheduled_start,
  scheduled_end, subject, teacher, year_group)` — one row per teaching period
  per ISO weekday, typed once on the classroom's Settings page (weekday tabs,
  "copy this day to Mon–Sat"). **Breaks are the gaps between rows**, so
  `has_following_period` and "when did the previous period end" are derived,
  not typed — which is what §Phase B′ was waiting for.
- `resolveSchedule(video, timetable, tz)` places a lesson in its week: the
  video's own bells win when present (an override, or a lesson from before the
  table); else the day's row named by the video's period label ("Period 3",
  "P3", "3" all match); else the row the recording's wall-clock window overlaps
  most (needs the anchor and the duration — i.e. no typing at all once the
  clock is known). The detail DTO exposes the result as `lesson.schedule` with
  its `source`, and both the analyze job and `/rederive` hand the ML service
  the period from it. Verified on the full handover recording with its bells
  cleared: `source: timetable`, same numbers (09:54, 4.1 min late), and the
  previous teacher's departure now reads **"30.6 min after her own 09:25 bell,
  25 of which was the break"** — her R3 against her own period, off the
  period-3 file, with no typing.
- The per-video timetable columns stay as the override and as the only source
  for lessons recorded before the table existed.
- Seeded for the handover classroom: the 7B bells (C.T + periods 1–8) for
  Monday–Friday, subjects and teachers still to be filled from the sheets.

Still to build:

- **Attendance register view**: one row per period per classroom-day, with the
  three-state arrival and departure, and the files that cover it. This is the
  page the school actually asked for; the per-video page becomes evidence
  behind it. Depends on Phase B (three states) and, across files, on Phase C.
- A lesson is **derived**: classroom × date × period. Not a row until someone
  overrides something on it (an attribution choice, a corrected bell time).

### Phase E — Anchor fallback: OCR the burned-in clock (small, shared)

Identical to Phase 5 of the attribution plan; listed here because coverage is
built on the anchor and this school's archive has none. Fixed crop, top-right
~470×56 px, read at t=0 and at t=end, accept only if the two agree with the
container duration to within a few seconds. Writes `recordingStartedAt` with
provenance `"ocr"`; a typed-in value still outranks it, as in
`scripts/backfill-recording-start.ts`.

---

## 8. Open decisions

1. **What does the camera actually do?** Fixed schedule aligned to bells, or
   continuous with fixed-interval splits, or a person pressing record? The
   45:00 duration and the 2-second lead say "scheduled"; that would make Phase
   C rarely needed and Phase B urgent. **Ask before building C.**
2. **What are the bell times?** The 4.5-minute figure assumes a 09:50 bell for
   period 3. Nobody has confirmed it. Every acceptance number in §9 moves with
   it.
3. **Lower bound or Not Observed when the file ends before she arrives?**
   Recommendation: lower bound, shown as "at least N minutes late", because it
   is true and actionable. The counter-argument is that "at least 45 minutes"
   on a period she never attended reads as a lateness rather than an absence;
   the register view should render it as **absent for the covered part of the
   period** when the bound equals the coverage length.
4. **Sibling window for Phase C.** 30 minutes either side of the period is a
   guess. A continuous recorder makes it irrelevant; scheduled files make it
   generous. Pick after decision 1.

---

## 9. Acceptance criteria

**Non-negotiable:** a file that starts on the bell, runs past the end bell and
contains one adult reports **exactly what it reports today**, state
`observed` on both ends. Pin `dto.test.ts`'s existing 09:49 fixture as that.

Fixtures to add, all pure DTO tests, no GPU:

| Case | Files | Expected |
| --- | --- | --- |
| Ends before end bell, she is present at end | 09:50 → 10:34, P2 = 10:35 | departure **lower bound**, not "1 min early" |
| Covers bell, empty file | 09:50 → 10:35, no presence | arrival **lower bound**, at least 45 min; register shows absent |
| Starts after bell, she is in frame at t=0 | 09:55 → 10:40 | arrival **not observed** ("no recording covers the 09:50 bell") — never "on time" |
| Two files, empty at bell, arrives in second | 09:10 → 09:55 (empty after 09:50), 09:55 → 10:40 (arrives 09:57) | arrival **observed** 09:57, 7 min late, `fileCount: 2` |
| Two files, adult present across the boundary at the bell | 09:10 → 09:52 (present to end), 09:52 → 10:40 (present from start) | **refused** with the boundary reason, until attribution lands |
| Rolling over midnight (trap 5) | 23:58 → 00:43 | `lessonDate` from the school-day, not the file start |

On the real recording, after Phases B and E: anchor recovered by OCR as
09:49:58, coverage 09:49:58 → 10:34:59, `coversScheduledStart: true`,
`coversScheduledEnd: false` if the bell is 10:35 — and therefore departure
**lower bound**, which is the correct reading of a file that stops one second
before the bell.

---

## 10. Suggested order

**A and B first**, same day: A is a document and two warnings, B is DTO
arithmetic with fixtures, and together they stop every false "left early" and
every erased lateness from the single-file case, which is probably the only
case this school has (decision 1). **E next**, because without an anchor
nothing in B ever fires on this archive. **Then decide C** on the answer to
decision 1. **D** when the school wants the register rather than the per-video
page, which is when they ask "show me the week".
