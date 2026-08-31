# LateAct workshop submission notes

## Chosen central claim

LateAct validates commitment-aware rollback efficacy and lower late-arrival
latency for Matrix-Game 2.0 mouse-yaw, while a separately frozen minWM
Wan2.1 Action2V yaw protocol independently replicates the underlying
denoising-time commitment phenomenon. The paper does not claim cross-model
rollback efficacy, cross-model latency savings, or universal action
generality.

## Paper story and structure

1. Motivate asynchronous action arrival as a serving problem omitted by
   fixed-action world-model evaluation.
2. Define a noise-coupled intervention and normalized response curve that
   identifies a denoising-time commitment boundary.
3. Introduce LateAct as exact rollback to the latest pre-commitment state,
   with an overwrite audit reducing the Matrix-Game checkpoint to 339,360
   bytes.
4. Establish the mechanism on Matrix-Game and independently on minWM yaw.
5. Validate rollback response, visual safeguards, and latency on a disjoint
   1,024-arrival Matrix-Game serving test.
6. Close with failure-preserving scope: the original minWM lateral gate
   remains failed, and keyboard rollback fails trajectory fidelity despite a
   shared commitment boundary.

## Results emphasized

- Matrix-Game mouse-yaw response improves from 0.218 to 0.997 with exact
  minimal rollback.
- On 32 fresh contexts, all 676 late arrivals are near restart, all 348 early
  arrivals use zero rollback, and mean late-arrival latency is 11.93% lower
  than restart.
- The optimized exact checkpoint is 339,360 bytes.
- The frozen minWM yaw confirmation passes 16/16 oracle-valid,
  sign-consistent, monotone, bounded direction-scene pairs.

## Results de-emphasized or moved to limitations/appendix

- The original minWM `a/d` curves are descriptive only because the frozen
  oracle-separation gate passed on 6/16 pairs.
- Keyboard shares the commitment boundary and recovers directional response,
  but its future-SSIM of 0.6726 fails the predeclared 0.90 safeguard; it is not
  evidence of action-general rollback fidelity.
- Per-scene minWM gaps, exact cache-byte accounting, full policy
  distributions, and detailed audit procedures are in the appendix.

## Anticipated reviewer concerns

- **Scope:** rollback efficacy is shown on one model/action family. The paper
  states this in the abstract, introduction, limitations, and conclusion.
- **Metric validity:** optical-flow evaluators can be action-mismatched. Both
  directions, independent affine/LK sign checks, oracle-separation gates,
  same-action controls, and the preserved `a/d` failure address this concern.
- **Trajectory fidelity:** high directional response need not reproduce the
  exact intended trajectory. The failed keyboard SSIM gate is retained as a
  central limitation rather than hidden.
- **Systems realism:** timings are RTX-3090- and implementation-specific. The
  paper reports measured action-to-pixel latency and exact checkpoint costs,
  not estimates from NFE counts alone.
- **Cross-model overclaiming:** minWM tests commitment only; no minWM rollback
  or latency result is claimed.

## Format check

The draft uses the official NeurIPS 2026 `dblblindworkshop` style with the
workshop title `World Models in Physical AI`. The compiled main paper is six
pages; references begin on page 7 and the appendix on page 8.
