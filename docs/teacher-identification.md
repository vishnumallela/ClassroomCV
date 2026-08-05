# Finding the teacher

Every KPI this product ships — her entries and exits, her time at the board,
her heatmap — hangs off one decision: which of the twenty-odd people in the
room is the teacher, and which box is her in each frame. This document records
how that decision is made, and why it is made this way.

## The failure it replaced

Measured on an 11.5-minute lesson from a real school (28 people, 1080p ceiling
camera, 208 raw BoT-SORT ids), the previous pipeline **labelled no teacher at
all**: `teacher_present_ms = 0`, every identity `unknown`, every dashboard
number zero.

The tracker was not the problem. It followed her almost perfectly, with three
long ids covering the whole lesson (0–208 s, 265–632 s, 632–692 s). The
problem was the order the questions were asked in:

1. `merge.py` grouped raw ids into identities by appearance and geometry, and
   put her three ids into **three different identities**.
2. `roles.assign_roles` then ranked whole identities and required the winner to
   beat the runner-up by a margin — but the runner-up **was also her**. Best
   0.845, second 0.749, lead 0.096, required 0.127. No teacher.
3. `teacher_chain` could then repair nothing, because it started from a teacher
   that did not exist.

Both symptoms reported from the field are this one bug: her id "changing" when
she leaves and comes back is her timeline being split across identities, and
"nothing in the first few minutes" is those early minutes belonging to an
identity that lost the vote.

## The approach

Solve her timeline **once, globally**, over tracklets, before identities exist.

### 1. Tracklets, not raw ids

A raw tracker id is a person only until the tracker hands it to somebody else.
Ids are split (`teacher_track.build_tracklets`) at three kinds of evidence:

| cut | what it catches |
|---|---|
| a long silent gap | the id came back on someone else |
| a sustained step in perspective-normalized height | the classroom steal: she crouches at a desk, her box collapses onto a pupil's, and the id walks away on him |
| a change of clothing in the timestamped CLIP gallery | the same steal between two people of the same height, which size cannot see |

Over-splitting is free — the assignment re-joins adjacent pieces of one person
at no cost — while under-splitting welds a pupil to the teacher permanently.

### 2. Age as the anchor, not behaviour

Behaviour is exactly what breaks in the reported cases: during the opening
minutes the whole class is standing, a pupil sent to the board walks and
stands, and a teacher sitting with a group stops looking like a teacher. She
is, however, the only adult in the room for the entire hour.

`app/adult.py` measures four things, all **relative to this video's own
population**, so there are no absolute pixel or colour constants anywhere:

- **stature** — height against a ground-plane model fitted across everyone
  (height vs the y of the feet). This is what stops the front-row child, who
  is the tallest box in the frame, from reading as the adult.
- **proportions** — head/torso and leg/torso from pose keypoints. Scale-free,
  so they survive distance, and they survive a desk hiding the lower body.
- **distinctiveness** — how far this person's torso colour sits from the
  room's own consensus. In a uniformed school that consensus *is* the uniform,
  and nobody had to name a colour.
- **zero-shot** — CLIP adult-vs-child prompts over the crops re-ID already
  embedded. Free when CLIP has run, absent otherwise.

Two properties matter more than the weights. Every measurement is weighted by
**how much data stands behind it**, and the total is **shrunk towards a child
prior**, so absence of evidence reads as "probably a pupil, like everyone else
in this room" rather than as a free pass. Without that, three-detection
fragments outranked the teacher.

Calibration note from real footage: in a senior class the teacher is only
about 10% taller than her pupils, so stature is evidence, not proof. The
head/torso ratio was dropped from scoring entirely — on a ceiling camera it
measures head tilt, and the measured adult/child values came out the opposite
way round from the anthropometry it was supposed to encode. A prior the data
contradicts is not a prior worth keeping.

### 3. One global choice

`teacher_track.select_timeline` runs weighted interval scheduling over
time-disjoint tracklets: value is `(score − threshold) × duration`, with a
penalty for physically implausible jumps between consecutive claims. Solved
exactly by DP in O(n²) over the few hundred tracklets a lesson produces.

This is what handles the cases a forward-chaining repair could not:

- **on camera from the first frame** — every tracklet is scored on its own
  merits, so there is no seed horizon to reach backwards through
- **leaves and returns minutes later at another door** — a gap costs nothing
  once it is long enough that her position carries no information
- **occluded over and over in a packed room** — fragmentation is expected; the
  DP simply resumes
- **two candidates alive at once** — impossible for one person, and
  disjointness rules it out by construction rather than by a tolerance

### 4. Iterated, but anchored

The seed is the most teacher-like *long* tracklet by behaviour and age — no
appearance involved, so appearance can never bootstrap itself. Her appearance
prototype is then built from the claims, and the pass repeats.

The prototype stays **anchored on the seed**, and a claim joins it only if it
genuinely resembles the seed (a robust-z outlier test). Rebuilding it from
"the highest-scoring claims" instead let one wrongly claimed pupil in, after
which his classmates scored well against a polluted prototype and the search
walked away from the teacher — a correct first round became a wrong fourth
one, and coverage fell from 95% to 78%.

### 5. Occlusion is measured, not ignored

`detector._occlusions` gives every detection a 0–1 occlusion from who is in
front of it (the ground-plane rule: lower feet are nearer) plus frame
truncation. Appearance crops are only sampled from clean views, so a packed
room stops embedding the occluder instead of the subject. Height and colour
are read only off clean views; **pose proportions are not**, because the
keypoints that are visible still belong to this person, and gating them as
strictly as height threw the age evidence away exactly when it was the only
evidence left.

## Results

| | before | after |
|---|---|---|
| coverage (frames where she is visible and correctly labelled) | 0% | 89.9% |
| purity (frames called her that were her) | — | 98.9% |
| id switches | — | 2 |
| time to first correct label | never | 112 s |
| re-acquisition after she leaves frame | — | 56% |

## Known limits

- **A completely static teacher** — never stands, never moves, never
  approaches the board, in a room where every pupil is equally still — is not
  found. The only evidence separating her is body proportions and clothing,
  and that does not clear the bar that keeps unsupervised rooms teacher-free.
  Covered by the `sitting_teacher` scenario, gated at zero, and flagged as the
  next piece of work.
- **A mid-track handoff recovers one side, not both.** The instant an id
  changed body is only known to within the spacing of the evidence that
  revealed it, so the halves overlap and the disjointness rule keeps the
  stronger one. Purity is what is gated there: it may never claim the pupil.
- **`/rederive` sees less than `/analyze`.** The database stores one median
  appearance vector per raw track, not the timestamped gallery, so a re-derive
  cannot split an id on a change of clothing. Everything degrades gracefully;
  nothing errors.

## How to check any of this

```
uv run python eval/run_eval.py                 # scenarios + every captured fixture
uv run python eval/run_eval.py --scenarios     # synthetic only, ~3 seconds, no fixtures needed
```

See `eval/README` notes in the module docstrings: `capture.py` records a real
video's detection stage once, `annotate_teacher.py` builds per-frame ground
truth from a cue the pipeline is not allowed to use, `metrics.py` turns the
two into identity numbers, and `scenarios.py` reproduces eight real classroom
failure modes with truth that is exact by construction.
