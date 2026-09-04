# Multi-teacher handover — diagnosis and fix plan

**Status:** Phases 0-3 built and committed; **verified on the full 45-minute recording 2026-09-04** (see the end of Phase 3 — it took one more rule, `resolve_swaps`). Phases 4-5 not started.
**Date:** diagnosed 2026-09-03; phases 0-1 landed 2026-09-03; full-recording run 2026-09-04
**Branch:** `feat/rfdetr-pipeline`

Phase 0 stops the wrong numbers reaching anyone: sustained co-presence is
measured, surfaced as a fourth `attribution` tier on the data-quality report,
and R1-R6 report Not Observed with a reason instead of a number. Phase 1 stops
the evidence being destroyed: migration 0014 makes `detection_events.track_no`
nullable and every teacher-class box is persisted, attributed or not.

Neither fixes the blend. The stored timeline for a handover lesson is still a
frame-by-frame mix of two people — it is now refused rather than reported, and
the boxes needed to unmix it are kept. Phases 2 and 3 are the fix.

Pick this up cold: everything needed to resume is below, including the evidence,
the exact code that is wrong, and the traps found on the way.

---

## 1. The problem in one paragraph

A 45-minute recording of period 3 starts while the **period 2 teacher is still
in the room**. She leaves a few minutes in; the period 3 teacher arrives. The
pipeline assumes a lesson contains exactly one teacher, so it silently blends
the two women into a single timeline and reports the *first* one's arrival as
the lesson teacher's arrival. The punctuality numbers are wrong, always in the
flattering direction, and the system grades its own confidence in them as
**high**.

---

## 2. Evidence (measured 2026-09-03, not inferred)

### The recording

| | |
| --- | --- |
| File | `~/Downloads/20082026/fixed/4957a8af-3d98-4c22-80ef-9d268b51afe3.mp4` |
| Duration | 2700.9 s (45:00) |
| Video | HEVC 1920x1080 @ 25 fps |
| Audio | AAC **16 kHz** mono — passes the transcription gate |
| `creation_time` | **absent** — both copies were written by ffmpeg (`encoder: Lavf62.x`), which stripped it |
| Burned-in clock | `2026-08-17 09:49:58` at t=0, top-right, ~470x56 px |

The folder is named `20082026` but the lesson is **17 August**. Folder name is
not the lesson date.

The burned-in clock advances exactly in step with the video timeline (checked at
t=0, 1350, 2698; the ~2 s slack is keyframe seek, not drift). **This is a better
anchor than `creation_time` ever was** — it is re-checkable at any offset.

### The handover, from the pixels

| Wall clock | Offset | What happens |
| --- | --- | --- |
| 09:49:58 | 0 s | Recording starts. **Cream-striped teacher** seated at the desk, marking. Class unsettled |
| 09:52:37 | 160 s | Cream teacher at the board with papers — still period 2 business |
| **09:54:25** | **267 s** | **Black-dressed teacher first appears** |
| 09:54:52 | 294 s | Both in frame together |
| ~09:55:30 | ~332 s | Cream teacher leaves |
| 09:55:00 → | 300 s → | Students seated facing front. Period 3 is running |

The two teachers wear **clearly different outfits** (black vs cream-striped).
An early read that they were hard to tell apart was wrong — it conflated the
cream-striped woman across two frames.

### What the pipeline produced

6-minute trim (`ffmpeg -t 360 -c copy`), run through the real pipeline on a GPU pod.

- classroom `b9172a20-d420-4507-b08b-540b5b58d80e`
- video `4d60bca1-fe6a-4d30-888e-fb80857c1b43`

| Field | Value |
| --- | --- |
| tracks | **1** — `track_no 1, teacher, conf 0.848, 0 → 359,880 ms` |
| `presence_intervals` | **`[[0, 359880]]`** — one unbroken interval, whole clip |
| `teacher_present_ms` | 359,880 (100%) |
| entries / exits | 1 / 0 |
| coverage | 0.913 |
| mean confidence | 0.782 |
| breaks / longest gap | 1 / 8,011 ms |
| `data_quality.confidence` | **`overall: high`, `coverage: high`, `continuity: high`, `teacher: high`** |
| notes | `"4 teacher detection(s) were rejected as implausible jumps"` |

So R1 = 09:49:58 → **on time**. Truth ≈ 09:54:25 → **~4.5 min late** (if the
period 3 bell is 09:50). No mid-lesson absence reported. Graded high confidence
on every axis.

### What the detector actually saw

The pipeline stores only boxes the single track claimed, so this had to be read
from `detect_video`'s raw output on the pod. Window: clip 250–340 s
(= wall 09:54:08 → 09:55:38), `sample_fps=5.0`, teacher conf ≥ 0.4.

```
raw detections (all classes): 1951
sampled instants with >=1 teacher box: 448
  1 box:  159 instants
  2 boxes: 287 instants
  3 boxes:   2 instants
instants with TWO OR MORE teachers: 289  (64%)

09:54:24.40   conf 0.85 x=0.558  |  conf 0.84 x=0.474
09:54:20.60   conf 0.81 x=0.556  |  conf 0.76 x=0.479
09:54:14.60   conf 0.82 x=0.501  |  conf 0.79 x=0.565
```

**RF-DETR sees both teachers perfectly.** Two bodies ~8% of frame width apart,
both at 0.7–0.86. The information is there and unambiguous. The pipeline
discards one.

### The failure is worse than "fusion"

Only **4** detections were rejected as implausible jumps — out of **289**
co-present instants. So 285 were not rejected at all:
`_pick_candidate` takes `max(reachable, key=conf)` each frame, and with two real
people both around 0.8 it picks whichever scores higher **on that frame**.

For 90 seconds the stored "teacher track" is a **frame-by-frame blend of two
people**, not teacher A followed by teacher B.

Consequences for this lesson:

- `presence_intervals` means "at least one adult was in the room"
- the heatmap is two women's dwell on one plate
- board / writing time could belong to either
- nothing is flagged, because continuity looks perfect — there is always *a* box

---

## 3. Root cause

Two different jobs are conflated under one name:

- **Detection** — which boxes are teachers? RF-DETR does this excellently.
- **Attribution** — which of those people is *the teacher this lesson assesses*?
  **Nothing does this job at all.**

`build_teacher_track` looks like it does the second but only ever did the first,
because with one adult they are the same question.

Stated outright at `services/ml-service/app/teacher.py:42`:

```python
# The teacher is one person, so she gets one identity number.
TEACHER_TRACK_NO = 1
```

### Where the assumption lives

| File / line | What it does | Why it is wrong |
| --- | --- | --- |
| `app/teacher.py:42` | `TEACHER_TRACK_NO = 1` | one track per video, by construction |
| `app/teacher.py:111-112` | `if gap_ms >= FREE_GAP_MS: return True` | any box after 5 s joins the chain — welds two people together |
| `app/teacher.py:120-138` | `_pick_candidate` → `max(reachable, key=conf)` | treats co-present people as competing guesses about one body; no record a competitor existed |
| `app/events.py:53` | `teacher_no = next(... role == "teacher")` | takes the first teacher track |
| `app/db.py:103` | `if d.track_no is not None` | **discards every unclaimed teacher box before it reaches the DB** |
| `app/quality.py` `assess()` | tiers coverage / continuity / confidence | has no notion of "more than one person", so a blend scores high |
| `apps/api-service/src/router/dto.ts:28` | `const first = presence[0]` | blindly takes the first interval, whoever it belonged to |

---

## 4. Recommendation: do NOT revamp the architecture

The queues, detector, shared clock, read-time derivation and rederive path are
all sound and none are implicated. There is one false assumption propagating
through four files. A rewrite carries all the risk and fixes nothing extra.

### Target shape

```
detections (all classes)
      |
 [A] multi-person tracking      -> N segments, not 1      (rewrite)
      |
 [B] persist ALL teacher boxes with segment id            (rule change)
      |
 [C] subject attribution        -> which segment is hers  (NEW STAGE)
      |
 [D] KPI derivation over the attributed segment only
      |
 [E] read-time R1-R6 + provenance / confidence
```

`detector.py`, the queues, the API surface, the frontend and the whole audio
path are untouched.

---

## 5. Phases

### Phase 0 — Stop reporting wrong numbers (small, do first) — BUILT

Count simultaneous teacher boxes. Above a threshold, mark the lesson
`multiple_adults_detected` and return **null** for R1–R6 with a reason, instead
of a number.

Fixes nothing; converts a confident wrong answer into a visible gap. These
numbers grade real teachers, so this should land before anything else. Fully
independent of every design question below.

Touches: `app/teacher.py` (count from the existing `by_ts` map),
`app/quality.py` (surface it), `dto.ts` `toPunctuality` (refuse).
`video_analytics.data_quality` is jsonb and additive — **no migration needed**.

### Phase 1 — Persist every teacher detection (small) — BUILT

`app/db.py:103` drops any box the single track did not claim. That is how
today's evidence was destroyed before it reached the database.

Storing all of them makes every later change to the attribution rule a free
`/rederive` over stored rows instead of a paid GPU re-run. Since the definitions
are still being argued out, this is the phase that pays for the rest.

Cost: ~20% more rows on a compressed hypertable (1951 raw vs 1641 stored on the
6-minute clip). Negligible.

> **Implementation trap found:** `detection_events.track_no` is **NOT NULL**
> (`apps/api-service/src/db/schema/index.ts`, `trackNo: integer("track_no").notNull()`).
> Storing unattributed boxes needs migration **0014** to drop that constraint,
> plus `fetch_detections` (`app/db.py:147-166`, `track_no=int(r["track_no"])`)
> handling `None`. Semantics: `NULL` = "detected as a teacher, not attributed to
> a tracked person". A sentinel like `0` avoids the migration but collides with
> `roles_map` keys — prefer nullable.
>
> **Hand-write migration 0014.** Do NOT run `drizzle-kit generate` blindly: it
> regenerates hand-written `IF NOT EXISTS` columns it cannot see in its snapshot
> (this broke 0012 against the live DB). Guard every statement.

### Phase 2 — Multi-person tracking (medium, the real work) — BUILT

`build_teacher_track` (one chain) becomes `build_teacher_tracks` (several).

- `_pick_candidate` stops treating simultaneous boxes as competing guesses about
  one body. **Two boxes at one instant is proof of two people**, not a tie to
  break. The module docstring already nominates this function as the place to
  add this "if a future room DOES show identity switching" — that room has now
  shown up.
- The 5-second free-gap rule stops silently welding chains. Segments may end and
  new ones begin; deciding whether two segments are the same person moves to
  Phase 3, where there is evidence to decide it with.

Reuse the existing motion model (`MAX_SPEED_PER_S = 0.8`, `JUMP_BASE = 0.05`) —
already measured against ground truth on two annotated lessons.

**No ByteTrack, no Re-ID encoder, no new dependency.** With 2–3 adults at 5 fps
on a static camera, distance-gated greedy association is sufficient. If a room
ever shows five adults, that is when a real tracker earns its place.

#### What landed (2026-09-03)

`build_teacher_track` now runs a greedy assignment tracker: per instant, drop
same-body duplicates, match boxes to active segments by **nearest reachable
box** (one box per segment), and a leftover box either welds onto the one
person unambiguously returning from an absence or starts a new segment. Output
is `.segments` (all), `.others` (substantial ones, numbered 2..), and an
**interim primary** — the biggest segment — which is exactly the old chain on a
one-adult lesson and is refused by Phase 0 on a two-adult one until Phase 3
replaces it. Other adults reach the API as `TrackOut.role = "adult"`
(`role_confidence: None`) with their own overlay; the player draws only
`teacher`, so nothing new renders yet.

**Verified on the real baseline, not just fixtures.** Phase 1 stored all
10,604 teacher boxes for the 37-min lesson, so the old chain and the new
tracker were run on the identical candidate set: **byte-identical primary**
(10,561 detections, same span, same mean confidence, 0 boxes differ, notes
identical) and identical `derive_result` output field for field, overlay
included. The synthetic handover splits into two lane-pure bodies with zero
swaps.

**Two things the real data taught, both now in the code:**

- **Dedup is containment, not IoU.** The first cut used IoU ≥ 0.5 and the
  baseline broke by 3 detections: the detector often puts a half-height box on
  her upper body beside the full-body box (same x, same top edge, IoU
  0.41–0.45), so those became two lanes and the phantom lane then collected a
  student across the room. The smaller box is 100% inside the larger; two
  people side by side are not. `SAME_BODY_OVERLAP = 0.7` on
  intersection/smaller-area.
- **Noise floor is 3 s / 8 detections.** A student called "teacher" for some
  frames is a segment but not a person — never numbered, never a weld
  candidate. Longest such run on the baseline: 0.8 s.

**Documented limitation, not fixed:** while she is briefly occluded, a student
box within reach of her last position joins HER segment (only continuation on
offer). The old chain did the same. Appearance (Phase 3) can catch it; motion
cannot. Pinned as a test so nobody mistakes it for a regression later.

#### Verified on the real handover (2026-09-03, fresh GPU pass, 2,048 boxes stored)

The card that read **"Teacher 100%, 1 in · 0 out"** now tracks five bodies,
refuses the punctuality numbers (`co_presence 78.8 s`, `attribution: low`),
and stores every box. What the segments say, checked frame by frame against
the video:

| segment | span | who |
| --- | --- | --- |
| T2, T3 | 0 → 68 s | the cream teacher **seated** — a head-only box and a torso box, two lanes (see below) |
| T1 | 65 → 360 s | the cream teacher from the moment she stands (primary by size; **swaps to the black teacher at ~307 s**, see below) |
| T4 | 247 → 262 s | the black teacher **in the doorway** — 09:54:05, twenty seconds before the note in §2 said she "first appears"; the detector is right |
| T5 | 262 → 336 s | the black teacher at the board; cream leaves at 335.8 s (09:55:34) |

Three things the real clip changed, all landed with the 37-minute baseline
still byte-identical:

- **Containment threshold 0.7 → 0.5.** On a seated teacher the model draws a
  head-only box beside a torso box; the head box hangs over the torso box's
  top edge, so containment is 0.5–0.9, not 1.0. At 0.7 she was two lanes for
  25 seconds.
- **Assignment by velocity-predicted position, not last position.** The two
  teachers cross paths twice. At the first crossing (≈258 s, black walking in
  from the door past cream at the board) nearest-to-last-position swapped
  them; predicting carries the walker through. **Fixed.**
- **A box-size term was tried and removed.** No benefit on either crossing,
  and the 308 s frame shows why it would mislead: cream's box legitimately
  grows to h≈0.37 as she walks toward the camera.

**Known limit, not fixed — the second crossing (≈306 s).** Cream walks left
past black and is undetected for a moment behind her; the model then draws a
partial box on the occluded teacher. A partial box's *centre* sits higher than
a full box's on the same standing person, so centre-distance matched each body
to the other's box. Tracking the box **top** (the head) instead resolves
exactly this crossing — and re-swaps the first, and **splits the 37-minute
baseline at 390 s**: a two-frame false box (a student ahead of her) seeds a
lane whose stale prediction eight seconds later sits closer to her next box
than her own lane's does, and steals it. Rejected on that evidence via
`tools/ab_tracker.py`. What resolves an occluded crossing is appearance,
which is Phase 3 — and the two outfits here (cream stripes vs black) are as
easy as appearance gets.

So after Phase 2 the segments are **pure between crossings, not across
them**, and fragmented (cream is three segments before she stands; black is
two). Phase 3's linking step has to do the joining; Phase 3's appearance step
has to catch the one remaining swap. The interim primary (biggest segment) is
cream for 65–307 s and black after — wrong either way, and Phase 0 keeps it off
the dashboard.

### Phase 3 — Attribution (medium) — BUILT

Pick the lesson's teacher from the segments, in this order:

1. **Overlap with the scheduled period `[P1, P2]`** — the primary rule.
   5 minutes vs 40 in this video. Uses the timetable fields already built.
2. **Segment duration + board/writing association** — she teaches, so she is at
   the board.
3. **Appearance** — only to *link* segments across a gap (did she step out and
   return, or is this someone new). Not the deciding vote: today's black vs
   cream is easy, next week's two similar saris will not be.
4. **Voice** — once the transcribe bug is fixed, an independent sensor agreeing
   with the pixels.

Output is a choice **plus a confidence and the reason**, and `undetermined` is a
permitted outcome that falls through to Phase 0's behaviour.

#### What landed (2026-09-03) — `app/appearance.py`, `app/attribution.py`

The order above survived contact with the data, with one correction: rule 1 is
not "overlap with the period" (on any recording that starts before the bell the
outgoing teacher has MORE overlap) but the observable fact behind it — **the
teacher whose period it is stays to the end, and the one who hands over leaves
while the other remains.** Presence within the period breaks ties among those
who stayed.

**Appearance** is a 16-float HSV histogram of the central 60% × 15–75% band of
each teacher box, computed at detection time on the frame the model just saw
and persisted in `detection_events.meta.app` (jsonb, additive; no migration),
so attribution re-derives from rows alone. Colour, not an embedding: no
dependency, and it is the *linker*, not the decider. Trimming the box sides
mattered — the students standing beside her wear green and yellow.

Three stages, each measured on both real lessons before any threshold was set:

1. **Split at a change of person.** Mean descriptor of the 30 boxes before an
   instant vs the 30 after; cut at ≥ 0.42. The handover's swap peaks at 0.48
   and 0.52; the 37-minute baseline never exceeds 0.355 (0.44 at window 20 —
   one occlusion event at 1034 s, which is why the window is 30).
2. **Link pieces into people** — each piece, in start order, joins the ended
   piece it most plausibly continues: appearance distance plus a position term
   that is decisive for gaps under 5 s and neutral past that. **Neither signal
   alone links the real clip.** A student in green stood in front of the black
   teacher for 40 s and her descriptor went 0.33 from her own other pieces
   (cross-person minimum is 0.21); continuity carries that seam (0.4 s, 0.046
   apart). Across the swap, position says the wrong thing (black stood still)
   and appearance says the right one (0.13 vs 0.34). A piece may speak for its
   appearance only with ≥ 8 clean boxes making ≥ 50% of it — the seated
   teacher's head-only lane manufactured a descriptor from 28 boxes and landed
   0.23 from the doorway piece, making her "arrival" t = 0. Duplicate lanes
   (a head box escaping the tracker's containment dedup) merge only when
   nested in the same **column**; "same place" folded the black teacher into
   the cream one for the 15 s they stood side by side.
3. **Attribute.** Handed over = final sighting followed by ≥ 10 s of another
   adult with you absent. High confidence needs ≥ 60 s of the remaining adult
   alone; the 6-minute trim leaves 24 s → **medium**, which the dashboard still
   withholds. The full recording leaves 39 minutes.

**Results.** Baseline: one person, 10,561 boxes, **identical KPIs, tiers and
notes** to the stored row; zero splits. Handover, re-derived through the app:
the card that read "Teacher 100%, 1 in · 0 out" now has `teacher 246.9 →
359.9 s` (the black teacher, from the doorway at 09:54:05), cream as an
"adult" 7.8 → 335.8 s, attribution **medium**, R1–R6 withheld with the reason
on the card: *"1 other adult left while this one remained; the adult who stayed
is attributed. She then held the room alone for 24 s — too little to grade her
on; the full recording would settle it."*

**Known limits, all on the non-chosen person:** the seated teacher's head-only
lane (0–33 s) survives as a phantom third adult, because the boy standing
beside her desk was detected as "teacher" for 25 s at x = 0.61 and the tracker
welded his lane onto her torso when she stood; her own arrival therefore reads
7.8 s instead of 0. Nothing in the attributed timeline is affected.

**The timetable still has to be typed in** for the handover lesson
(`period_known: false` today): recording start 09:49:58 from the burned-in
clock, lesson 2026-08-17, period 3 09:50–10:35 if that is the bell. With it the
API passes `period_start_ms/period_end_ms` on both `/analyze` and `/rederive`.

**Rows analysed before Phase 3 have no descriptors**; attribution then links by
continuity alone and its reason says so. The two lessons here were backfilled
locally from the videos. New GPU runs compute descriptors natively.

#### Verified on the full 45-minute recording — 2026-09-04

The complete period-3 file (`760713c7`, 2700.9 s, 13,690 teacher-class boxes
with descriptors, RTX A4000 in EUR-IS-1 at $0.25/hr, 17 min detection —
decode-bound at 100-280% CPU, GPU at 2-7%) was the first real test of Phase 3
beyond the 6-minute trim, and Phase 3 as committed got it **wrong**: it chose
a 10.8-minute chimera (the cream teacher's first 5.5 minutes plus the black
teacher's last 5.5) at *medium*, and reported the black teacher — present for
35 minutes — as having handed over. Nothing wrong reached the dashboard (the
refusal held), but the answer was not there either.

What actually happened, read off the stored rows (`tools/ab_tracker.py`-style
replay, frames with boxes drawn at the instants in question):

1. **A third adult.** A colleague in a white shirt walked in at 2373.6 s
   (10:29:31), passed *behind* the black teacher at 2381 s while she was
   briefly undetected, and sat at the left desk until the end. The motion
   model did exactly what §5 Phase 2 says it does at an occluded crossing: the
   teacher's 35-minute lane followed the colleague to the desk, and the
   teacher's remaining five minutes became a new lane.
2. **The change-point split could not cut it.** Black against white scores
   0.33 under the HSV descriptor (two of its three parts are near-identical
   for unsaturated clothing, so the distance saturates around 0.4), and the
   37-minute single-teacher baseline reaches 0.36 on its own through an
   ordinary occlusion. No threshold separates them. Where the *other* lane did
   split (0.49 — white torso then black), the adjacent link re-joined the two
   pieces on position alone.
3. **So the black teacher's tail could not rejoin her**: her own lane was
   still alive on the colleague's body and overlapped it by 68 s.

**The fix is `attribution.resolve_swaps`** — step 0, before the split. At any
instant two substantial lanes are within reach of each other, compare both
lanes' appearance windows (6 s) on either side of the instant, as tracked and
re-paired: `d(Xb,Xa)+d(Yb,Ya)` against `d(Xb,Ya)+d(Yb,Xa)`. Measured at the
2381 s swap: 0.77 as tracked against 0.36 re-paired; swap when the re-pairing
wins by `SWAP_MARGIN = 0.25`. The same test at the *clean* crossing (velocity
carried the walker through) reads cream|cream + black|black against the cross
terms, so it leaves that alone; a lane with no second substantial lane beside
it is never examined, so the baseline is untouched by construction; a student
recolouring one lane's window raises one cross term, not both. The 306 s
crossing on both real lessons is now also resolved as a swap (score ≈ 0.7)
instead of by split-and-relink, with the same outcome. Two swaps on the full
recording, one on the trim, none on the baseline.

Two smaller rules came out of the same run:

- **Confidence by lead.** The colleague stopped being detected 12 s before the
  end, so "held the room alone" read 12 s and the call was capped at medium.
  A candidate present `LEAD_HIGH` (2×) longer than anyone who left is now also
  *high*; the trim (113 s against 336 s) stays medium, as it should.
- **`Candidate.left_ms`.** The colleague's pieces linked to the cream teacher's
  by appearance across a 34-minute gap (cream stripes vs a white shirt: 0.16
  — same-person pieces sit at 0.10-0.25, so no threshold separates these
  either), which would have put the period-2 teacher's departure at 10:32
  instead of 09:55. An adult's *departure* is now the end of her presence run
  containing the bell (absences under `LEFT_BRIDGE_MS` = 5 min bridged), and
  `dto.ts` reports that rather than her last sighting.

Result through the app (`analysis.rederive`, no GPU):

| | Before | After |
| --- | --- | --- |
| Attributed | chimera, 7.8 → 2700.8 s, 647 s present | black teacher, 246.7 → 2700.8 s, 2354 s present |
| Confidence | medium (withheld) | **high** |
| Coverage | 22% | 91% |
| R1 arrival | withheld | **09:54, 4.1 min late** |
| R3 departure | withheld | 10:34, at the bell |
| Presence share of period | withheld | 87.6% |
| Previous teacher | "left 09:50, 0.5 min in" (the head-only lane) | **left 09:55, 5.6 min into the period** |

Still true and still documented: the seated teacher's head-only lane (0-33 s)
is a phantom second adult at the bell (`adultsAtBell: 2`); the colleague's
seated minutes are three descriptor-less pieces, each "an adult who left".
Neither moves a number.

The descriptor's ceiling is the finding to carry forward: black vs white and
cream vs white are both inside the same-person range. `resolve_swaps` needs
only the *pair's* consistency, so it survives that; anything that needs an
absolute appearance threshold (the split, long-gap links) does not, and a
better descriptor (a value-weighted or joint histogram, or a torso band that
does not move with box height — torso-only boxes score 0.33 against
full-body boxes of the same person) is the next lever if a third real lesson
breaks either.

### Phase 4 — Review queue (small)

When attribution is uncertain, show the detected people as thumbnail crops and
let someone click the right one. Stored as an override that outranks the
automatic choice — same precedent as a typed-in value outranking a container tag
in `scripts/backfill-recording-start.ts`.

Not data entry for every lesson; a queue that is usually empty. **Only worth
building if someone will actually work it** — otherwise refusing is the whole
answer.

### Phase 5 — Two loose ends (small, independent)

- **OCR the burned-in clock** as a fallback for `recording_started_at` when the
  container tag is missing. Fixed crop, top-right ~470x56. Unblocks Group A for
  this school's entire archive.
- ~~**Fix `transcribe.ts`** (see §7)~~ — done. Re-run the audio.

---

## 6. Acceptance criteria

**Non-negotiable: the single-teacher case must produce identical output.** The
37-minute lesson at 94.5% coverage is the validated baseline and this change
must not move it.

> ~~**Problem:** `detection_events` was wiped for the older videos, so there is
> no stored fixture to replay against.~~ **DONE 2026-09-03.**

### The frozen baseline

Video `b6d19a9c-45c6-4ad7-b8b3-0fe0129c3543` (`test_video1`, 37.3 min,
2560x1440 @25fps), re-run on a fresh pod with Phase 0+1 in the image. It
reproduces the pre-change reference EXACTLY, which is the point:

| | 2026-08-24 reference | 2026-09-03, with Phase 0+1 |
| --- | --- | --- |
| sampled frames | 11,179 | **11,179** |
| teacher tracks | 1 | **1** |
| teacher detections | 10,561 | **10,561** |
| span | 0 -> 2,234.6 s | **0 -> 2,234.6 s** |
| coverage | 94.5% | **94.47%** |
| mean confidence | 0.86 | **0.855** |
| overall quality | medium | **medium** |

Not "close enough" — identical detection counts and span. The single-teacher
case did not move.

**Two things this measured that synthetic fixtures could not:**

- **The co-presence threshold has real headroom.** This genuinely
  single-teacher lesson still reports `co_presence_ms: 8600` and
  `max_simultaneous_adults: 2` — the detector really does offer a second
  teacher box at 43 instants across 37 minutes. It was correctly NOT flagged
  (`attribution: high`), so `CO_PRESENCE_MIN_MS = 30_000` clears real
  double-detection noise by ~3.5x. That constant was a judgement call before
  this run; it is now measured against a real room.
- **Phase 1's storage cost is 0.41%, not ~20%.** 10,604 rows stored: 10,561
  attributed plus **43 unattributed** — exactly the losing boxes at those 43
  contested instants (43 x 200 ms = the 8,600 ms of co-presence, which is a
  clean internal consistency check on both numbers). The earlier ~20% estimate
  came from comparing raw all-class detections against the stored chain and was
  measuring the wrong thing.

Zones were auto-placed this run, so board time is real (5.6 min) where the
2026-08-24 run reported null. Presence 2,196 s across 4 intervals, 2 entries /
1 exit, 10 timeline breaks (longest 17 s) -> continuity `medium`, which is what
drags overall off `high`.

> **Cost that run:** the configured `RTX PRO 4000 Blackwell` had NO CAPACITY in
> EU-RO-1 (four attempts, no pod created, nothing billed); `RTX PRO 4500
> Blackwell` came up first try at $0.72/hr. The first analysis then died on the
> DATA_DIR bug now fixed in `dc9cb9d`. Budget two attempts on a fresh pod.

On the handover clip, after the fix:

- two segments, not one
- attribution picks the black-dressed teacher
- R1 = 09:54:25, R2 ≈ 4.5 min late
- `presence_intervals` covers only her

---

## 7. Separate bug: audio never ran — FIXED

The audio job failed at submit:

```
transcript submit failed: {"error": "`language_detection` is not available when `language_codes` is specified."}
```

`apps/api-service/src/lib/transcribe.ts` sent **both** `language_codes:
["en","hi"]` and `language_detection: true`, which the API rejects at submit. It
was also missing two settings proven necessary on real classroom audio.

Nothing was billed — it fails before upload.

| Change | Why |
| --- | --- |
| **drop** `language_detection` | mutually exclusive with `language_codes`, and `language_codes` is the one to keep — see the correction below |
| **add** `speech_models: ["universal-3-5-pro"]` | without it the account default runs: one speaker, key terms ignored. Note plural + array; `speech_model` singular is deprecated |
| **add** `speaker_options: {min_speakers_expected: 2, max_speakers_expected: 6}` | `speaker_labels` alone returned the whole 4.5-min lesson as ONE speaker. With the floor: 2 speakers. Mutually exclusive with `speakers_expected` |

> **CORRECTION to this section's original advice.** It said to drop
> `language_codes` and keep `language_detection`, on the grounds that detection
> "reports which languages were heard, which R21 needs". That is wrong on both
> counts, checked against AssemblyAI's docs before applying it:
>
> - `language_detection` resolves the FILE to a single dominant language. On a
>   Hinglish lesson it picks a winner and mangles the other half — precisely the
>   failure `language_codes` was added to prevent, and one already verified on a
>   real lesson.
> - It reports **no per-utterance language** for pre-recorded audio, so it never
>   could have served R21.
>
> `language_codes` is the code-switching parameter (max 2, one must be `en`) and
> Universal 3.5 Pro supports mid-sentence Hinglish. R21's per-turn language has
> to come from the turn's own text, which code switching returns in its native
> script — the same place the Devanagari/Latin normalisation already has to
> happen before anything is counted.

`transcribe.test.ts` pins each of these against a stubbed fetch, because three of
the four fail SILENTLY rather than loudly: omit `speech_models` and the account
default answers with a 200 and a plausible transcript that has one speaker and
no key terms.

---

## 8. Open decisions (unanswered)

1. **When attribution is uncertain — refuse, or best-guess with a flag?**
   Recommendation: refuse. A gap prompts someone to look; a flagged guess gets
   read as a number and the flag gets ignored. Costs: some lessons report
   nothing.
2. **Phase 4 now or later?** Only if someone will work the queue.
3. **Re-analyse the full 45-minute video, or keep testing on trims?** Trims are
   cheaper to iterate on; the full run is the honest end-to-end test.

---

## 9. Environment notes for whoever resumes

### Reproducing the GPU run

```bash
# 1. API service (port 8787, not 3011 — that is the frontend)
cd apps/api-service && bun src/index.ts

# 2. Create the pod from the app, NOT the RunPod console
curl -s -X POST http://127.0.0.1:8787/rpc/gpu/create \
  -H 'content-type: application/json' -d '{}'

# 3. Upload the checkpoint (no network volume -> /workspace dies with the pod)
scp -P <port> ~/Desktop/classroomcv-models/rfdetr-medium-mt2vr2m9__checkpoint_best_total.pth \
  root@<ip>:/workspace/weights/rfdetr-medium.pth        # ~34 s

# 4. Reverse tunnels (MinIO 9000, Postgres 5533) — leave running
services/ml-service/tools/pod_tunnel.sh

# 5. Verify FROM INSIDE the pod, not from the Mac (no curl in the image; use python3)

# 6. When done
curl -s -X POST http://127.0.0.1:8787/rpc/gpu/terminate -H 'content-type: application/json' -d '{}'
```

### What it cost / what worked

- **NVIDIA L4 was unavailable** in EU-RO-1 with CUDA pinned to 13.0.
  **RTX PRO 4000 Blackwell** worked — $0.57/hr, driver 580.159.04, 24 GB.
- Pod `3osp7fti9dcxe7`, ~14 minutes, **~$0.13 total**.
- 6-minute clip analysed in ~2.5 min.

### Traps hit today

- **A stale local ml-service runs on `127.0.0.1:8000` on `device: mps`.** With no
  pod, `mlServiceUrl()` (`app-settings.ts:265`) falls back to it, so an upload
  **silently analyses on the laptop** instead of failing. Check
  `mlServiceUrlEffective` before trusting a run.
- The ml-service image has **no `curl`** — verify tunnels with `python3`.
- sshd sessions do **not** inherit the container env. Pass explicitly:
  `RFDETR_WEIGHTS=... DEVICE=cuda REQUIRE_DEVICE=cuda RFDETR_BATCH=16 RFDETR_RESOLUTION=576 DATA_DIR=/workspace/data`
- The app dir on the pod is `/srv/ml-service`, not `/app`.
- The pod does **not** keep the analysed video (`/workspace/data` is empty after
  a run) — upload a clip separately for any raw-detector probe.
- This ffmpeg build has **no `drawtext` filter** — contact sheets cannot be
  labelled in-filter.
- Do **not** run repo-wide `bun run format` — it rewrites all 32 files including
  eval fixtures. Format only what you changed.

### The raw-detection probe

Saved at `services/ml-service/probe.py` on the pod (now terminated). It calls
`detect_video` directly and histograms teacher boxes per sampled instant. Worth
re-creating as a committed tool — it is the only way to see what the detector
saw before tracking throws it away.

---

## 10. Suggested order

Phases **0 and 1** first: small, low-risk, independent of every open decision,
and together they stop the wrong numbers reaching anyone and make everything
after them free to iterate on. Then the baseline re-run (§6), then Phase 2.

**Done: 0, 1, the §6 baseline, 2, the real-handover pass, and 3.** Next: type
the handover's timetable in, then the honest end-to-end — the FULL 45-minute
recording on the GPU, where the handover leaves 39 minutes of the period-3
teacher alone and attribution should reach **high**, and R1 should finally
read 09:54:05 rather than 09:49:58. Then Phase 5's loose ends.

Two things learned while building them, worth knowing before Phase 2:

- **`DataQualityOut` in `app/models.py` is a hard gate.** Pydantic defaults to
  `extra="ignore"`, so a key added to `quality.assess()` without a matching field
  there is dropped silently by `AnalysisResult.model_validate` — it reaches
  neither the API nor the jsonb column, and nothing raises. The column being
  schemaless jsonb makes this easier to miss, not harder. `TrackOut.role` is
  `Literal["teacher"]` and will need widening the moment Phase 3 labels a
  segment anything else.
- **A pre-0014 lesson cannot be rescued by `/rederive`.** Its stored rows are the
  winning chain only, so a replay sees one box per instant and finds no
  co-presence. Those lessons need a fresh detector pass, not a re-derive.
