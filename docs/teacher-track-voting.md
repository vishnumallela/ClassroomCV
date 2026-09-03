# Multi-track voting for the teacher

**Status: designed, not implemented.** Written 2026-08-26 after a false
detection on `test-final-RFDETR` (1:46). Nothing in this document has been
measured against the eval harness yet; the numbers quoted are from existing
docs and from reading the code, not from a run of this design.

## The failure this exists for

At ~1:18 of `test-final-RFDETR` the `Teacher` box sits on a **child in school
uniform** presenting at the board. The **adult** — standing right of the board
by the cupboard — has no box at all.

That is a detector error, not an association error, and it matters that the
distinction is clear before choosing a fix:

- The tracker only ever chooses among boxes the detector emitted. It cannot
  invent a box on the woman the model missed.
- The child stands roughly where the teacher usually stands. At 5 fps the
  motion gate allows `0.05 + 0.8 × 0.2 ≈ 0.21` frame-fractions of movement
  (`JUMP_BASE`, `MAX_SPEED_PER_S` in `app/teacher.py`), and the gap between the
  two bodies is well inside that. So the switch passes the gate silently and is
  **not** counted in `rejected_jumps`.

### Why not ReID / ByteTrack

Considered and rejected — see also `storage-and-reid-design.md`.

1. **There is no person detector any more.** `EXPECTED_CLASS_NAMES` is
   `["Door", "Screen", "Teacher", "pointing", "writing"]`. ReID embeds person
   crops; the only crops available are the ones already labelled `Teacher`,
   i.e. the wrong ones. Adding ReID means first re-adding a person detector
   (`ultralytics`), an embedding model (`onnxruntime`/OSNet) and an assignment
   solver (`lap`) — approximately the 2,750 lines and three heavyweight deps
   that 36f5792 deleted.
2. **ReID answers the wrong question.** It answers "is this the same body?",
   not "which body is the teacher?". Fed a wrong anchor it propagates the wrong
   answer with *higher* confidence across the lesson. Coverage and `mean_conf`
   both rise while the answer gets worse — the error becomes invisible in the
   metrics.
3. **Cost.** Rough estimate, unmeasured: a second full detection pass plus one
   embedding per person box (~30 children per frame) is 2–4× wall-clock on top
   of the current 37 min → 10.9 min.
4. **~30 children in identical white uniforms** at this resolution, mostly
   seated and mutually occluded, is close to the worst case for appearance
   embeddings.

What the ReID proposal was really reaching for is **lesson-level voting**, and
that can be had without any of the above.

## The design

Three phases, reusing the existing primitives in `app/teacher.py`. All of it is
post-processing on detections already in memory — no extra GPU pass.

### 1. Separate — greedy multi-chain

Instead of one chain from one seed:

```
remaining = teacher dets >= threshold
while remaining:
    seed  = first uncontested instant in remaining
    chain = _chain(remaining, seed)
    fragments.append(chain)
    remaining -= chain
```

`_plausible` and `_chain` are reused unchanged.

> **Disable `FREE_GAP_MS` during this phase.** This is the subtlety that makes
> or breaks the whole approach. The 5 s free-gap rule makes a chain accept
> *anything* once the gap is large enough, so with it on, one chain swallows
> both people and you get no separation at all. Strict motion gate here;
> bridging comes later.

### 2. Vote

Score each fragment by **Σconf**, with detection count as tie-break. For the
failure above the adult spans the lesson and the child spans about a minute, so
count, span and Σconf all rank correctly — Σconf is preferred only because it
folds presence and certainty into one number.

**Discard rule — overlap, not global ranking.**

> Drop a fragment only when it **temporally overlaps** a stronger fragment.

"Most present wins" globally is too blunt. If the child's fragment coexists with
the adult's, the child loses on evidence and that is exactly right. If the
child's fragment is the *only* thing in that window — the adult genuinely
undetected — a global rule would still delete it, sometimes correctly and
sometimes just blanking the overlay. The overlap rule makes each discard
justified rather than statistical.

### 3. Bridge

Re-apply the 5 s free-gap rule *within the winner only*, so her genuine exits
and re-entries still work. This is the behaviour `FREE_GAP_MS` was written for
and it must survive.

## Constraint: `track_no` is the privacy gate

**Do not persist the losing fragments.**

`db.py` writes a detection iff `d.track_no is not None` — that single condition
is the entire filter, and the module header states the invariant:

> ONLY THE TEACHER IS STORED. […] "students are never displayed" is a property
> of the data rather than of the renderer — there is no student box in the
> database to leak into an overlay by mistake.

Assigning `track_no = 2, 3, 4…` to candidate fragments would write **boxes of
children** into the database — the candidates are false positives, so that is
the premise, not an edge case. On children's classroom footage that is a real
regression, and it quietly downgrades a data-level guarantee to a
renderer-level one.

So: **vote in memory, persist only the winner as `track_no = 1`.** Full
benefit, no privacy change, no schema change.

### The cost of that choice

`/rederive` replays teacher-only rows out of `detection_events`. If voting
happens in-memory at `/analyze` time, the losing fragments are gone and the
voting policy **cannot be retuned later without re-running the detector**.

The alternative — persisting candidates so the policy can be replayed — is
exactly what the invariant forbids. Recommendation: keep the invariant and tune
against the offline harness instead.

## What this does not fix

**A missed detection.** Voting selects among detected candidates. If the adult
has no box at all in a window, no association method invents one — the same
wall ReID hits. Before building this, run the diagnostic:

```
RFDETR_WEIGHTS=... uv run python run_one.py <video>
```

and inspect the detections at ~1:18 **below the 0.4 cut**.

| finding | what it means | fix |
|---|---|---|
| she scores ~0.3, child scores ~0.5 | ranking problem | this document |
| she has no box at 0.15 at all | detection problem | retrain with hard negatives |

Hard negatives, if it is the second: children presenting at the board, adults
facing away, adults occluded by furniture. The model has likely learned "person
standing at the board" as a teacher cue, which is the shortcut this frame
exposes. Note there is no training tooling in this repo — that checkpoint was
produced elsewhere.

## Validation

Confined to `app/teacher.py`, which is a pure function of detections, so
`eval/run_eval.py` scores old-vs-new against the khaitan and demo ground truth
before anything ships. Implement behind a flag and report the coverage/purity
delta.

Expect a smaller delta than the screenshot suggests, for a reason worth
knowing: **the current single-chain code already self-heals after 5 s.** When
`_pick_candidate` returns `None` the chain does not update `prev`, so the gap
grows until `FREE_GAP_MS` opens the gate and the chain re-attaches. A latch onto
the wrong body is bounded, not lesson-long.

The genuinely silent failure — and the one this design targets — is the child
standing *inside the jump allowance* and winning on confidence. That switch is
invisible in `rejected_jumps` today.

Two further cautions on measurement:

- The 95.7% / 97.4% figures in `rfdetr-pipeline.md` come from khaitan and demo.
  If `test-final-RFDETR` is a third room it is **unmeasured**, and a single
  frame does not establish whether this is a one-minute artifact or systematic.
  Annotating a few hundred frames of it would.
- Per the standing note, `detection_events` was wiped while tracks/analytics
  survived, so nothing can be measured against stored rows — re-run.

## Open questions

- Does the adult score anything at the 0.15 floor at ~1:18? Everything above
  branches on this.
- After separation, can fragments of the *same* person across a >5 s gap be
  re-joined without appearance? Position carries no information there by
  design, which is the gap the current code deliberately punts on.
- Should a winning fragment shorter than some fraction of the lesson be
  reported as "no teacher found" rather than accepted?
