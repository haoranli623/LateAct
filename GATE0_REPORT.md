# LateAct Gate 0 report

## Verdict

**B. COMMITMENT-CURVE STRONG GO**

Matrix-Game 2.0 exhibits a reproducible within-block action commitment curve.
With three denoiser evaluations, a mouse-yaw action arriving after evaluation 1
is expressed almost exactly like the NEW oracle (`p1` median 0.997). Waiting
until after evaluation 2 sharply reduces incorporation (`p2` median 0.218).
This timing effect is monotone in all 16 direction-scene pairs and survives both
left-to-right and right-to-left action reversal.

This is not free late binding: only 2/16 cases retained `p2 >= 0.8`, below the
frozen 12/16 A threshold. It is also not immediate hard commitment: no case had
`p1 <= 0.2`. The predeclared B criterion passes.

No Gate 1, training, repair, model edit, or rollback experiment was started.

## Executed design

- Substrate: official Matrix-Game 2.0 1.8B checkout at
  `71c3cd7f741311f8100f6cf9cde942b6c1378d11`
- Official universal scenes: 0000 through 0007, fixed before the smoke
- Prefix: one generated three-latent block
- Tested future: one three-latent block (12 decoded-frame equivalent)
- Directions: mouse-left (`yaw=-0.1`) to mouse-right (`yaw=+0.1`) and reverse
- Switch positions: `s=0,1,2,3`; OLD for the first `s` denoiser calls, NEW for
  the remainder, with no reset of `noisy_input`
- Oracles: `s=0` is NEW throughout; `s=3` is OLD throughout
- Sampler: BF16, three NFEs at 1000.0, 908.8427124023438, 713.9794311523438
- Context write: one additional official `context_noise=0` call after the
  output is fixed, explicitly excluded from NFE and scientific timing position
- Unique generated futures: 6 per scene (two oracles plus four interior
  direction/switch combinations), 48 total

## Exact Matrix-Game action/cache hook

The official `cond_current` slices `mouse_cond` and `keyboard_cond` for each
denoiser call (`pipeline/causal_inference.py:110-127,293-330`). Each active
transformer block calls `ActionModule` after visual cross-attention
(`wan/modules/causal_model.py:273-288`). Universal configuration enables the
action module in blocks 0-14; blocks 15-29 have no action module
(`configs/distilled_model/universal/config.json:4-5` and
`wan/modules/causal_model.py:205-208`).

The hook altered by LateAct is the per-call `conditional_dict`, not a model
parameter or foundation-model source file. At each NFE, the wrapper selects the
OLD or NEW `mouse_cond` immediately before calling `pipeline.generator`.
`keyboard_cond`, `cond_concat`, `visual_context`, timestep schedule, model
precision, and branch-start cache state are fixed. The latent and current-block
cache slices then evolve causally without any mid-trajectory restore or swap.

Official mouse cache addressing is at `wan/modules/action_module.py:277-304`;
keyboard uses the analogous path at lines 405-431. With the one-block prefix,
active action-cache indices begin at `(global,local)=(3,3)` and visual indices
at `(2640,2640)`. The first future NFE advances them to `(6,6)` and
`(5280,5280)`. Later NFEs retain those endpoints and overwrite only the current
block positions `[3:6]` (visual `[2640:5280]`). The prefix positions `[0:3]`
and `[0:2640]` were asserted bit-identical after every NFE and after the context
write. Inactive action layers correctly remain `(0,0)`.

## Coupling and isolation audit

Each scene materializes three BF16 CPU tensors before branching: one initial
block-noise tensor and two scheduler re-noising tensors. Every run reuses their
exact SHA-256 values; no global RNG call occurs in the branching loop. For
scene 0000, the hashes are:

- initial: `af601ae2eda85648327179213b1c403c1f192b81dccd1261a744afe363596a4e`
- re-noise 1: `26ac22b8662576990b5a81a50c4bd701292f642df5ad4af3c8e1973b48556dd4`
- re-noise 2: `87e3dc8702fd866793f9b88c18e941efa9f84c4efd8c8622fe8eab81854eb74f`

Before every branch, the full visual, mouse, keyboard, and cross-attention
caches are restored from one CPU prefix snapshot. Dynamic fail-closed guards
confirmed all of the following for every smoke and Gate 0 run:

- the only differing condition tensor is `mouse_cond`;
- all three explicit noise hashes match within a scene;
- visual-prefix, mouse-history, and keyboard-history K/V remain exact;
- cross-attention K/V and its initialization state remain exact;
- cache indices follow the expected append-once, overwrite-current behavior;
- each trajectory has exactly 3 denoiser evaluations and 1 separate context
  write.

The smoke's four same-action switch schedules per scene produced identical
latent SHA-256 values and an exact signed-motion range of 0. This is the measured
BF16 repeatability/numerical floor.

## Metric

The primary signed camera score is the sum of robust median horizontal
Lucas-Kanade feature-track displacements over every transition from the final
prefix frame through the future block, at 320x176. Per-transition tracks are
MAD-filtered; median oracle support ranged from 107 to 589 tracks. Dense
Farneback median horizontal flow was frozen as an evaluator-stability check.
Its oracle direction agreed with the primary metric in all 16 cases.

For OLD score `y_old`, NEW score `y_new`, and switch score `y_s`, the reported
new-action fraction is `(y_s-y_old)/(y_new-y_old)`, without clipping. Oracle
signed gaps were large and stable: absolute mean 114.27 pixels of accumulated
track displacement, range 109.29-122.02. All 16 pairs were evaluator-valid.

## Results

| Scene | LR p1 | LR p2 | RL p1 | RL p2 |
|---|---:|---:|---:|---:|
| 0000 | 0.968 | 0.179 | 0.981 | 0.289 |
| 0001 | 0.996 | 0.178 | 1.000 | 0.438 |
| 0002 | 1.007 | 0.248 | 1.010 | 0.136 |
| 0003 | 0.997 | 0.061 | 0.994 | 0.038 |
| 0004 | 0.997 | 0.497 | 1.006 | 0.507 |
| 0005 | 1.003 | 0.245 | 0.997 | 0.191 |
| 0006 | 0.994 | 0.932 | 1.006 | 0.823 |
| 0007 | 0.997 | 0.127 | 1.001 | 0.077 |

Aggregate:

- `p1`: mean 0.9971, median 0.9971, range 0.9680-1.0097
- `p2`: mean 0.3104, median 0.2182, range 0.0381-0.9321
- paired `p1-p2`: mean 0.6867, median 0.7740
- bootstrap 95% CI for mean paired effect: [0.5496, 0.8026]
- paired one-sided Wilcoxon p-value: 1.5259e-5
- monotone within frozen tolerance: 16/16
- bounded intermediate responses: 16/16
- A/free-late-binding count (`p2 >= 0.8`): 2/16
- C/hard-after-first-NFE count (`p1 <= 0.2`): 0/16

Scene 0006 is a real qualitative/quantitative exception in which the final NFE
still redirects most motion. It was retained and is why the conclusion is a
distributional commitment curve rather than a universal hard boundary.

## Runtime, VRAM, and state bytes

All reported generation timings include three NFEs, the official clean-context
write, and the exact cache guards. They exclude VAE decoding.

- mean block generation: 1.2274 s (median 1.2211; range 1.2118-1.4620)
- throughput: 2.444 denoiser NFE/s; 9.777 decoded-frame-equivalent/s
- mean VAE decode: 1.1824 s per six-latent prefix+future video
- total branch GPU time: 58.915 s
- total prefix GPU time: 9.418 s
- two-GPU artifact wall span after report creation began: 119.3 s
- peak allocated VRAM/GPU: 10,065,501,184 bytes (9.374 GiB)

Per-fork allocated cache storage:

| Cache | Bytes |
|---|---:|
| visual self-attention | 973,210,080 |
| mouse action | 648,806,880 |
| keyboard action | 737,760 |
| visual cross-attention | 47,370,240 |
| total | 1,670,124,960 |

These are allocated BF16 buffers, including inactive capacity; they are not a
compression result.

## Qualitative branches and artifacts

The scene-0000 montage (`gate0_qualitative_montage.png`) shows endpoint oracles
and both directional sequences. Visually, `s=1` remains near the NEW endpoint,
whereas `s=2` is generally much nearer OLD, matching the signed motion scores.
All 48 videos, future latents, cache/noise traces, and per-scene metrics are in:

`/mnt/NAS/data/hl5757/generated_artifacts/lateact/gate0-20260823`

Primary machine-readable files:

- `gate0_summary.json`
- `gate0_shard_0.json`
- `gate0_shard_1.json`
- `gate0_qualitative_montage.png`

Smoke artifacts and the exact negative-control floor are in:

`/mnt/NAS/data/hl5757/generated_artifacts/lateact/smoke-20260823`

## Recommendation and bounded next step

The hidden phenomenon exists: action binding is nearly free after the first
evaluation but usually collapses between the second and third. Recommend
reviewing Gate 0 as **B. COMMITMENT-CURVE STRONG GO**.

If a later Gate 1 is approved, the minimal repair hypothesis is a one-evaluation
rollback: retain the pre-evaluation-2 latent and current-block cache slice, and
on action arrival after evaluation 2 restore only that boundary state before
replaying evaluation 2 onward with the new action and the same stored re-noise.
This is a proposal only; it was not implemented or tested here.
