# LateAct final polish notes

## 1. Matrix-Game calibration statistics

The one-sided Wilcoxon p-value over 16 direction pairs was removed from the
main commitment-calibration result. The paper now reports only the descriptive
evidence: 16/16 valid pairs, median responses `0.997` and `0.218` after one and
two OLD evaluations, and the paired median drop `0.774`.

## 2. Preregistration-like wording

Every requested manuscript occurrence was tightened:

- Abstract: “predeclared trajectory-fidelity safeguard” became
  “pre-specified trajectory-fidelity safeguard.”
- Experimental design: “predeclared limitation study” became
  “pre-specified limitation study.”
- minWM appendix: the external-preregistration disclaimer was replaced by an
  explicit statement that the chronology is adaptive and the follow-up
  protocol was fixed before confirmatory execution.

The paper continues to state accurately that relevant thresholds and
protocols were frozen before their confirmatory runs.

## 3. Motion-estimator wording

The second LK/partial-affine estimate is now described as a “complementary
sign-consistency check,” not an independent sign check, because both estimates
derive from the same video evidence.

## 4. Discussion scope addition

A compact “Deployment scope” paragraph now states that full LateAct efficacy
is limited to Matrix-Game mouse-yaw; minWM tests only commitment; arrivals are
controlled/stratified rather than production-sampled; measured absolute
latency remains about `2.28 s`; the boundary is calibrated rather than
dynamically predicted; and no physical-control objective or robot is tested.

## 5. Serving statistics preserved

The n=8 base-image-cluster serving inference remains unchanged:

- response mean/median `0.909136 / 0.930385`, 95% CI
  `[0.859769, 0.954093]`, 8/8 positive, one-sided exact p=`0.00390625`;
- latency mean/median `0.311789 / 0.313654 s`, 95% CI
  `[0.305519, 0.317523]`, 8/8 positive;
- mean late-arrival latency reduction `11.93%`.

The n=64 cells remain descriptive quality units, not independent inferential
samples. Mouse-yaw SSIM-tail and temporal-safeguard values are unchanged.

## 6. Scientific claims and results

No frozen result, threshold, artifact, figure, central claim, or scope boundary
was changed. minWM remains mechanism replication only; original `a/d` remains
failed at 6/16; keyboard retains the shared-boundary, recovered-direction, and
failed trajectory-fidelity interpretation.

## 7. Final page count and checks

The compiled PDF contains 18 pages including references, appendix, and the
required checklist. The main paper remains six pages, and references begin on
page 7. Compilation has no unresolved citations, overfull boxes, broken
figures/tables, identity leaks, or public project/repository URLs. Only benign
underfull-box/template warnings remain.

## 8. Remaining reviewer vulnerability

The principal limitation remains external validity: full policy efficacy is
shown on one model/action family with eight independent serving base images,
controlled arrival schedules, action-specific evaluators, and
implementation-specific latency. The calibrated boundary is not yet an online
predictor, and neither a production controller workload nor physical task is
evaluated.
