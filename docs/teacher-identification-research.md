# What the literature says about the problem we are solving

A 2026 research pass over six areas — classroom video analytics, multi-object
tracking, person re-identification, age estimation, global data association,
and evaluation — read against the seven weaknesses measured in
`docs/teacher-identification.md`. Recorded because several of the findings are
NEGATIVE results that would otherwise be rediscovered expensively, and because
two of them contradict what the code assumed.

## The findings that changed the code

### Raw CLIP is not a weak re-ID model; it is a random one

Published zero-shot person-retrieval mAP for the exact encoder we were using:
CLIP ViT-B/32 scores **0.10 mAP on MSMT17** and 0.37 on Market-1501 — which is
approximately what random ranking returns. Scaling does not rescue it
(B/16 = 0.11, L/14 = 0.14), and self-supervised foundation backbones fail the
same way (DINOv2-B/14 = 0.37 MSMT17). Landmark retrieval and person
re-identification are not the same task, and no general-purpose encoder is a
re-ID encoder.

This exactly matches what we measured on our own footage before reading any of
it: different people sat at cosine 0.82–0.89 while two views of the teacher
reached 0.96. That is not a preprocessing bug to be tuned away — it is a model
operating as designed, on a task it was never trained for.

Two consequences, both acted on:

- **We tried the encoder swap and it lost.** The ultralytics-native person
  re-ID model (`yolo26s-reid.onnx`, 28 MB, no new licence) does separate raw
  crops far better in absolute terms — different people at cosine 0.12–0.23
  versus CLIP's 0.81–0.89. But absolute margin is not the question; ranking
  is. Measured end to end, twice, the second time with 448-pixel input on
  uncapped crops and batched inference so the encoder had its best case:

  | | embedding separation | coverage | purity |
  |---|---|---|---|
  | CLIP, khaitan | **85.7%** | **91.4%** | **98.0%** |
  | re-ID, khaitan | 58.6% | 37.9% | 75.1% |
  | CLIP, demo | **72.4%** | 76.5% | 83.7% |
  | re-ID, demo | 58.1% | 71.0% | 87.0% |

  The likely reason is domain, and it cuts the opposite way to the benchmark:
  MSMT17 crops are upright, street-level, full-body pedestrians; ours are
  40–200 px, top-down, half-occluded, framed on the upper body. A metric
  trained to tell pedestrians apart does not transfer to that, while CLIP's
  broad semantic features — clothing, texture, the sliver of scene around the
  shoulders — happen to. It stays wired up behind `REID_MODEL` for the
  experiment that is still open (carry re-ID, CLIP and colour as three
  modalities and let the fusion weigh them), but it is not the default.

  Note the order of operations that made this measurable at all: before the
  sampler was fixed, CLIP's own separation was far lower, and any encoder
  comparison run then would have measured the sampler, not the encoder.
- **A per-video metric-learning fix does not work.** We tried the obvious
  trick from the FACT line of work: mine negatives for free (two tracklets
  alive at the same instant are certainly different people), then whiten the
  embedding along the directions those negatives already differ in. Measured
  on both lessons it made discrimination *worse* (AUC 0.552 → 0.452). You
  cannot whiten information into an embedding that never had any.

Note also what NOT to reach for: supervised re-ID checkpoints overfit their
training camera network catastrophically — OSNet scores 83.6 mAP on
Market-1501 and **3.4 on MSMT17, 0.9 on PKU-ReID**. The BoxMOT default
`osnet_x1_0_market1501.pt` is precisely the wrong weight for an unseen ceiling
camera. If a heavier encoder is ever wanted, the cross-domain evidence favours
CLIP-ReID (MSMT17-trained, MIT) or SOLIDER (MIT), not a Market-trained CNN.

### Body proportions cannot separate a female teacher from teenage pupils

The anthropometric channel we invested in is, for the hardest case, provably
empty rather than merely hard. CDC NHANES III: female stature is **flat from
age 15 (64.1 in) to adult 30–39 (64.3 in)**; sitting height 85.7 cm at 16 vs
86.7 cm adult; biacromial breadth 36.7 cm vs 37.0 cm. Effect sizes d ≈ 0.0–0.25.
No calibration, 3D lifting or SMPL fitting recovers a signal the population
does not contain. The same measurements separate a *male* teacher from
13-year-old boys at d ≈ 2.4 on shoulder breadth.

This is why our measured stature advantage on a senior class was only ~10%, and
it is the reason the age term is one weighted, reliability-scaled signal rather
than the rule. It also predicts that the term will be much stronger on primary
classes than on the secondary lessons we have. Head size is reportedly the one
body proportion that retains signal — our head/torso ratio measured the
*opposite* of the anthropometric expectation on a ceiling camera and was
dropped, which is consistent with it measuring head tilt rather than head size.

### Over-splitting is not free

The published offline tracklet-association work that most resembles our stage
(GTA-link, ACCV 2024, MIT) refuses to split any tracklet shorter than 100
boxes, and its ablation reports that the **connector supplies most of the gain
while the splitter adds little**. We had assumed the opposite — "over-splitting
costs the assignment nothing" — and that assumption cost us: on the held-out
second lesson the splitter shredded the teacher's 136-second track into
nineteen pieces, destroying the behaviour evidence the search bootstraps from,
and the pipeline claimed two pupils. Fixed by requiring a height change to
arrive as an abrupt step and by refusing to split on height at all when the
perspective model is too weak to make "size" mean anything.

## Findings we have not yet acted on

| Finding | Source | Why it matters here |
|---|---|---|
| Offline association is *faster* as well as better: the same model at subclip T=256 runs 340 FPS vs 12 FPS online, and gains +4.6 HOTA / +7.0 IDF1 | NOOUGAT, arXiv 2509.02111 (IJCV 2026) | Validates the offline-batch premise; the horizon is a free accuracy knob we have not swept |
| Appearance matters *most* at long range — removing it costs −3.9 IDF1 at hierarchy levels covering 128–512 frames, and almost nothing at short range | SUSHI ablation, arXiv 2212.03038 | Our whole problem is long-range association, so the encoder swap should pay disproportionately here |
| k-reciprocal re-ranking is offline-only and large: +14.1 mAP on MSMT17 for SOLIDER-Tiny | SOLIDER-REID | We are offline and never pay this. Applies directly to the tracklet affinity matrix |
| Constrained clustering with cannot-link on temporal overlap is the standard formulation for exactly this ("which tracklets are one person") | video face-clustering literature | Our one-vs-rest DP is a special case; the general form would also solve the two-adults and TA cases |
| Lifted edges encode long-range consistency without changing the feasible set | Hornakova et al., ICML 2020 | Our DP scores only *adjacent* transitions; a claim consistent with the whole timeline is not rewarded |
| Freeze appearance updates while two tracklets overlap heavily | FC-Track, arXiv 2603.12758 | We gate sampling on occlusion, but not on inter-*track* overlap specifically |
| MiVOLO (Apache-2.0) does age estimation with **no face**, body branch trained with face dropout | AIST 2023 | The one obtainable, permissively-licensed way to get a real age signal; gate it behind a measurement on our own crops first |
| SAM 2.1 (Apache-2.0, weights included) propagates a single target bidirectionally from one confident anchor | Meta | The only single-target option that does not depend on fixing the embedding first; a natural fit for "click the teacher once" |

## The harness critique we agree with

An independent read of our own harness found that **39% of the pipeline's
teacher labels are never scored by any metric**, because ground truth abstains
on frames where she is not standing — so `cold_start_ms` can report 0 ms and
pass its gate while the product-visible cold start is tens of seconds, and the
static-teacher weakness is unmeasurable by construction. Two of the nine
counted "re-entries" are real; the rest are holes the annotator dug.

Partly addressed: `abstention` is now reported alongside coverage and purity,
so being silent and being wrong are no longer the same number. Not yet
addressed: ground truth should be track-level human identity labels rather than
a colour oracle, and the annotation-noise band should be measured (a second
pass over a random 10%) before any further purity improvement is claimed —
model-assisted labels have been shown to *beat* human MOT17 ground truth, which
means our 98% purity is plausibly inside our own labelling noise.
