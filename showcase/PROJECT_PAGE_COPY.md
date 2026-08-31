# Title

LateAct: Commitment-Aware Asynchronous Control for Interactive Video World Models

## One-sentence hook

LateAct restores the latest pre-commitment denoising state when an action arrives late, then recomputes only the necessary suffix to recover control without paying for a full restart.

## 3 headline metrics

- **0.218 → 0.997** median normalized late-action response
- **11.93%** mean post-arrival latency reduction versus full restart
- **339,360 B** exact rollback checkpoint (~339 KB)

## What problem does LateAct solve?

Interactive video world models can receive a user's action while a frame is already being denoised. Once the current trajectory has committed to the old action, directly swapping in the new action may leave the output following stale control. Restarting from the beginning recovers control, but discards useful computation and increases response latency.

## Key observation

In the frozen Matrix-Game 2.0 mouse-yaw evaluation, normalized response to the new action stayed near 1 after one old-action function evaluation, then fell sharply after the second: approximately 0.997 to 0.218 across 16 validated scene-direction pairs. This measured transition identifies a denoising-time commitment boundary: action conditioning remains plastic before it, but direct late binding is largely ineffective after it.

## How LateAct works

LateAct keeps a small exact checkpoint at the latest state before commitment. If a new action arrives after that boundary, it restores the checkpoint and recomputes only the affected suffix under the new action. This preserves reusable prefix computation while recovering behavior close to the new-action oracle and full restart.

## Results

On Matrix-Game 2.0 mouse-yaw, Gate 1 improved median normalized late-action response from 0.218 for direct late binding to 0.997 with minimal rollback; all 16 validated pairs improved and were near full restart. A fresh asynchronous Gate 2 evaluation covered 676 valid late-arrival cases and measured an 11.93% mean latency reduction versus full restart, with a 339,360-byte exact rollback checkpoint. The primary frozen evaluation also passed its response, quality, state-audit, and latency criteria.

## Scope / limitations

The strongest efficacy evidence is specific to Matrix-Game 2.0 mouse-yaw control. Keyboard control shares the measured commitment boundary but fails the frozen future-SSIM trajectory-fidelity requirement, so it is not evidence of action-general efficacy. minWM shows commitment-like exploratory behavior, but its formal independent-model replication gate failed (6/16 valid pairs); LateAct is therefore not claimed as cross-model replicated or universal.

## Buttons / links

- **Technical Report** — link to `GATE2_REPORT.md` (with Gate 0, Gate 1, Phase 3, and Phase 4 reports available as supporting evidence)
- **Code** — link to the repository root
- **Demo** — link to `showcase/lateact_demo.mp4`

## Source note

The three display metrics are copied without reinterpretation from the frozen Gate 1 and Gate 2 summaries. Exact values and source fields are recorded in [`headline_metrics.json`](headline_metrics.json).
