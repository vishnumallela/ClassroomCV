# The RF-DETR pipeline

The ML service runs one detector and follows one person. This document records
what it does, why every earlier stage was removed, and the measurements that
set each threshold.

```
Video
  │
  ▼  RF-DETR (fine-tuned, 5 classes)        app/detector.py
  │     door · screen · teacher · pointing · writing
  │
  ▼  gate the STATIC classes to their zones  app/zones.py
  │     the teacher is never gated — she has the run of the room
  │
  ▼  follow the teacher                      app/teacher.py
  │     continuity first, confidence second
  │
  ▼  the four KPIs                           app/heuristics.py → app/events.py
  │     entries/exits · board time · heatmap · pointing/writing
  │
  ▼  store HER detections only                app/db.py
```

## What went away, and why it could

The previous pipeline detected *people* and then had to work out which of the
thirty in the room was the teacher. That question is genuinely hard, and it was
answered with a ground-plane perspective model, an age estimator built from
stature and limb proportions, a torso-colour uniform-outlier score, a CLIP
appearance prototype, a dynamic program over tracklets, and a bounded
vision-model vote as a tiebreak — about 2,750 lines whose entire job was
inferring a label.

A detector trained on `teacher` as a class answers it directly. So all of it is
gone: `adult.py`, `merge.py`, `teacher_track.py`, `teacher_id.py`, `roles.py`,
`board_detect.py`, the BoT-SORT tracker config, and with them the `ultralytics`,
`openai-clip`, `onnxruntime` and `lap` dependencies.

**No Re-ID, and no ByteTrack/BoT-SORT.** Not as a simplification to revisit
later — the measurements say there is nothing for them to do. Re-identification
exists to tell one person from another after an interruption; there is no
competing identity here, because students are not a class the model detects. A
multi-object tracker exists to maintain many simultaneous identities; there is
one, and at the configured threshold the model emits **at most one teacher box
per frame** (0 frames with two, over 583 scored frames). What remains is gap
bridging and a plausible-motion check, which is what `app/teacher.py` is.

If a future room does show identity switching, the single place to add
discrimination is `teacher._pick_candidate`, where competing boxes already meet.

## Measurements

Against per-frame ground truth on the demo lesson (4:46, ~30 pupils, teacher in
a dark kurta) — the room the *previous* pipeline found hardest, and the one that
exposed its tuning as room-specific:

| | previous pipeline | RF-DETR |
|---|---|---|
| coverage | 76.5% | **95.7%** |
| purity | 83.7% | **97.4%** |
| id switches | — | 3 (all annotation artifacts — see below) |
| cold start | — | 0 s |
| re-entry recall | — | 100% |
| frames with no teacher box | — | 1.4% |
| frames with **two** teacher boxes | — | **0** |
| box IoU (p50 / mean) | — | 0.888 / 0.850 |
| longest unbroken miss | — | ~5.4 s |
| derive time | 0.2 s | 0.01 s |

On the 37-minute lesson, teacher presence holds at 94.5% of sampled frames.

### The remaining 2.6% is the annotator, not the pipeline

All 25 disagreeing frames were dumped and three were pixel-verified against the
video. Every one is an anchor whose box is a sliver under 9% of the frame width
pinned to the exact edge (`x=0.000` or `x+w=1.000`) — where the colour cue that
produced this ground truth cannot support a judgement:

- **149.0 s** — the anchor sits on a student's shoulder at the right edge. The
  teacher is standing mid-frame among the desks, and the pipeline detects her
  correctly at 0.9 confidence. The harness scores this as a miss *and* an id
  switch.
- **215.2 s** — the anchor sits on a student at the left edge. The teacher has
  stepped out of shot entirely; the pipeline correctly emits nothing, and the
  harness scores that as a coverage failure.

So the true accuracy is **higher** than the table records, and `id_switches=3`
is a count of annotation artifacts. This is the same limitation the previous
pipeline's own review recorded — the colour oracle digs holes, and measured
purity sits inside its own labelling noise. Gates are therefore set just under
the measured values as a regression tripwire, and **the last two points should
not be tuned against**. The real fix is re-annotating with track-level human
labels.

### Threshold: 0.4

Detect once, then score many thresholds off the identical boxes — re-running
detection per arm would measure detection noise as well as the knob:

| threshold | coverage | no box | box off target | **two boxes** |
|---|---|---|---|---|
| 0.15 | 95.7% | 0.0% | 4.3% | 8.1% |
| 0.20 | 95.7% | 0.7% | 3.6% | 0.5% |
| 0.25 | 95.7% | 1.4% | 2.9% | 0.0% |
| **0.40** | **95.7%** | **1.7%** | **2.6%** | **0.0%** |
| 0.60 | 95.7% | 1.7% | 2.6% | 0.0% |
| 0.70 | 95.2% | 3.4% | 1.4% | 0.0% |

Coverage is flat across a wide plateau, so the threshold was chosen for
*distance from both edges* rather than for a best score: 0.4 sits in the middle,
past the point where competing boxes disappear and well before 0.70 starts
losing her.

### Plausible motion: 0.8 frame-fractions/second

Measured from the ground truth itself, not guessed. Her frame-to-frame speed
peaks at 0.55 (demo) and 0.25 (khaitan) frame-fractions per second, so the gate
clears the faster room's maximum with headroom while still rejecting a box that
crosses the room between two samples. Past `FREE_GAP_MS` (5 s) the gate switches
off entirely: she may have left and re-entered by another door, so position
carries no information and must not veto her return.

### Zones

The board and door do not move, and the detector finds them almost every frame:

| | present in | centre x (p10–p90) |
|---|---|---|
| screen | 96.5–100% of frames | 0.527–0.529 |
| door | 77–81% of frames | 0.328–0.329 |

A positional spread of ±0.002 over an entire lesson is what makes zoning a
median rather than a search, and it is why `board_detect.py`'s open-vocabulary +
SAM 2 proposal chain (773 lines) is now ~80 lines in `app/zones.py`.

Zones do two opposite jobs: **propose** a polygon for a room on first upload,
and **gate** later screen/door detections to it so a wall poster that reads as a
screen for six frames cannot move the board.

## Teacher-only, at the data layer

Only the teacher's detections are written to `detection_events`. Students are
not a class the model detects, the board and door live in `zones` as one polygon
each rather than one row per frame, and `pointing`/`writing` are consumed during
derivation. So "students are never displayed" is a property of the stored data,
not a filter in the renderer — there is no student box in the database that a
future overlay change could leak.

## The action KPIs (pointing / writing)

The detector emits `pointing` and `writing` as their own classes, not as an
attribute of the teacher box, so two rules turn them into a KPI
(`heuristics.action_samples`):

**Attribution.** An action box is hers when its centre lies inside her box for
that frame, grown by `ACTION_EXPAND` (5% of frame). Centre-in-box rather than
IoU because it is correct whether the annotator drew the action on her whole
body or on the hand alone, and there is no per-frame ground truth for these two
classes yet to choose between those on evidence.

**Sampling.** One sample per TEACHER detection, not per action box. That makes
the series the same shape as the board's, so the same hysteresis machine
(`intervals_from_samples`) applies, and action time is a subset of presence time
by construction — `writing_ms` can never exceed `present_ms`.

Because only her boxes are persisted, `/rederive` cannot recompute these: it
replays `detection_events`, which never contained an action box. It therefore
returns `None`, and `replaceDerived` carries the previously measured value
forward rather than overwriting it — correct, because the action KPIs do not
depend on zones and a zone edit is the only reason to rederive.

`None` and `0` are kept distinct all the way to the tile: `None` renders as
"not scored yet", `0` as a measured zero.

**Not yet calibrated.** `ACTION_ON_MS` / `ACTION_OFF_MS` / `ACTION_FLICKER_SAMPLES`
are the only knobs in `heuristics.py` not set from a measurement — the action
classes have no annotated spans in `eval/gt`. Annotate a lesson and sweep them
the way `BOARD_ON_MS` was before trusting the absolute numbers.

## Testing

- `tests/test_teacher.py`, `tests/test_zones.py` — the per-rule behaviour
  (contested frames, gaps, re-entry, jump rejection, zone gating). Millisecond
  runtime, and they run in CI.
- `eval/run_eval.py` — the real path (`detect_video` + `derive_result`) over a
  real lesson, scored against per-frame ground truth in `eval/gt/`. Needs the
  video and the checkpoint, so it is a before-you-ship pass rather than a CI
  gate.

The harness this replaced replayed *frozen detection fixtures*, which made sense
when the detector was fixed and expensive and all the interesting logic sat
downstream of it. Now the detector is the interesting part, and a fixture of its
output could not catch a regression in it.

## Configuration

Everything tunable is in `app/config.py` and overridable by environment
variable:

| setting | default | what it is |
|---|---|---|
| `RFDETR_WEIGHTS` | *(unset)* | Path to the checkpoint. Unset = `/analyze` fails loudly; there is no second detector to degrade to. |
| `RFDETR_RESOLUTION` | 576 | The trained-at resolution. Not a free recall knob — changing it rescales every box the model learned. |
| `RFDETR_BATCH` | 8 | Frames per `predict()`. The main GPU throughput lever, and on cuda also the batch the fp16 JIT trace is built for; forced to 1 off-GPU. |
| `RFDETR_TENSORRT` | false | Serve as a TensorRT engine (cuda + `--extra tensorrt`). Off until parity-checked on the target GPU — see below. |
| `TEACHER_CONF` | 0.4 | See the sweep above. |
| `ZONE_CONF` | 0.5 | Door/screen. Held higher than the teacher: a false positive would move a zone. |
| `ACTION_CONF` | 0.5 | `pointing`/`writing`. Held above the 0.15 detect floor: a KPI measured in seconds must not be built from boxes the model barely believes. |
| `DEVICE` / `REQUIRE_DEVICE` | auto / *(unset)* | `REQUIRE_DEVICE=cuda` makes a mis-provisioned pod die at load rather than bill ~20× the wall-clock on CPU. |

Zone polygons are **not** configured here. They live per classroom in the
database and are drawn in the zone editor, which is where a per-room camera
layout belongs — one polygon per room, seeded automatically from the first
upload's detections and correctable by hand.

## The class-id contract

The checkpoint's label order is the entire contract between the model and every
KPI:

```
0 Door    1 Screen    2 Teacher    3 pointing    4 writing
```

`detector._check_class_order` reads the order back out of the checkpoint at load
and refuses to serve one that disagrees, because an off-by-one here would
silently report the door as the teacher rather than crash. The ids live in
`app/models.py` so light consumers can name a class without importing torch.

Note that the model's `Screen` is the product's `board`: the database, the API
contract and the zone editor all say "board", and `zones.ZONE_CLASS` is the one
place the two vocabularies meet.

## GPU precision and TensorRT

On cuda the model is JIT-traced in **fp16 at `RFDETR_BATCH`**. Both rfdetr
defaults are wrong for a pod — `dtype=float32` and `batch_size=1` — and an
untuned load runs at half speed and double the VRAM on a graph specialised for
a batch it never receives. `detector._optimize` handles it and tests pin it.

That ordering is deliberate. The previous pipeline adopted TensorRT on a "~5x"
claim; measured, it was 1.05–1.25x, and the actual win was a warmup call
silently pinning the backend to fp32. **fp16 is most of the win and is free.**

TensorRT itself is wired but **off by default**, for a reason worth stating
plainly: rfdetr ships TensorRT *export* but not TensorRT *inference* — its own
`_tensorrt` module points at a separate library for serving. So
`app/tensorrt_backend.py` is the missing half. It borrows rather than
reimplements (preprocessing reads `means`/`stds` off the loaded model; decoding
calls rfdetr's own `PostProcess`), but the resize, the `/255` and the batch
plumbing are still ours, and a mismatch there does not raise — it shifts every
box, and every KPI with it.

Hence `tools/trt_parity.py`, which must be run on the pod before the flag goes
on. It compares teacher-class detections between backends at the production
threshold and reports the real speedup **against fp16**, so the last mistake
cannot repeat. `GET /health` reports the backend that actually loaded, not the
one that was requested; every failure in the TensorRT path degrades to fp16
PyTorch and logs why.

The engine is compiled for one GPU model, one TensorRT version and one input
resolution — cached as `<weight>.r<resolution>.trt` on the volume, never baked
into the image.
