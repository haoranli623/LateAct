# Gate 2 mouse-yaw robustness analysis

This is a read-only analysis of the frozen `mouse_shard_*.json` rows. No model
generation, artifact mutation, threshold change, or sample exclusion was performed.

## Verified hierarchy

- 8 base images (`0008`--`0015`)
- 4 deterministic variants per base (`v0`--`v3`)
- 2 directions per variant
- 16 arrivals per direction
- 32 scene-variants, 64 directions, and 1,024 arrivals total
- Frozen late definition: `inflight_nfe >= 2`; 676 late arrivals retained
- Arrival classes: NFE1=348, NFE2=338, NFE3=338
- Quality safeguard unit: one decoded comparison per scene-variant/direction (n=64), not per arrival

## Clustered serving effects

Primary aggregation averages unchanged late arrivals within each direction,
then gives equal weight to the eight variant/direction cells within a base image.
The eight base images are the inferential units. The percentile bootstrap resamples
8 base images with replacement for 200,000 replicates using seed
20260829. Exact Wilcoxon signed-rank tests enumerate all 2^8 sign assignments.

### Response: LATEACT minus DIRECT-LATE-BIND

- Published direction-level reference (fixed `direct_late` branch, descriptive n=64):
  mean 0.815919, median 0.838861,
  direction-bootstrap 95% CI [0.768992, 0.858529].
- Important estimand note: that 0.8159 statistic is the stored fixed per-direction
  `lateact - direct_late` branch contrast. It is not the mean over all 676 late-arrival rows.
- Raw late-row direction-level descriptive estimate (n=64 direction cells): mean
  0.909136, median 0.931032.
- Base-image inference on all frozen late rows (n=8): mean 0.909136,
  median 0.930385, cluster-bootstrap mean 95% CI
  [0.859769, 0.954093],
  exact one-sided Wilcoxon p=0.00390625;
  8/8 base images positive.
- Paper-compatible fixed-branch n=8 estimate: mean 0.815919,
  median 0.857617, cluster-bootstrap mean 95% CI
  [0.714153, 0.908183],
  exact one-sided Wilcoxon p=0.00390625;
  8/8 positive.

### Latency: FULL-RESTART minus LATEACT

- Original direction-level reference (descriptive n=64): mean
  0.311789 s, median 0.307275 s,
  direction-bootstrap 95% CI [0.307107, 0.316612] s.
- Base-image inference (n=8): mean 0.311789 s, median
  0.313654 s, cluster-bootstrap mean 95% CI
  [0.305519, 0.317523] s,
  exact one-sided Wilcoxon p=0.00390625;
  8/8 base images positive.

Both qualitative conclusions remain unchanged under n=8 inference.

### Per-base-image aggregates

| Base | Late rows | Raw-late response gain | Fixed-branch response gain | Latency saving (s) | SSIM median | SSIM minimum | SSIM <0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0008 | 85 | 0.874301 | 0.743689 | 0.296872 | 0.929414 | 0.919560 | 0/8 |
| 0009 | 83 | 0.979654 | 0.959611 | 0.300060 | 0.904617 | 0.836061 | 3/8 |
| 0010 | 82 | 0.812422 | 0.622074 | 0.323829 | 0.960722 | 0.918649 | 0/8 |
| 0011 | 83 | 0.917893 | 0.833007 | 0.317065 | 0.935387 | 0.900379 | 0/8 |
| 0012 | 84 | 0.797887 | 0.583328 | 0.309640 | 0.933413 | 0.915010 | 0/8 |
| 0013 | 88 | 0.984340 | 0.970605 | 0.312523 | 0.875472 | 0.718375 | 6/8 |
| 0014 | 86 | 0.942877 | 0.882228 | 0.314784 | 0.895560 | 0.860178 | 4/8 |
| 0015 | 85 | 0.963713 | 0.932813 | 0.319535 | 0.926790 | 0.807984 | 2/8 |

### Existing analyzer compatibility note

The frozen `analyze_gate2.py` `paired_effect` field averages LATEACT-minus-DIRECT
over all 1,024 valid arrival rows, including the 348 early rows where the two
policies are identical. It is therefore not a late-arrival inferential statistic.
The published 0.8159 statistic instead comes from the 64 stored fixed
`lateact - direct_late` direction contrasts. Neither existing file was changed;
this robustness output uses raw shard rows and reports both estimands explicitly.

## Future-SSIM tail (frozen n=64 direction-level quality units)

Mean 0.915660; median 0.922305; sample SD
0.042420; minimum 0.718375;
p05 0.839679; p10 0.874391; p25 0.900347;
p75 0.946301; p90 0.958383; p95 0.960936;
maximum 0.973400.

- SSIM < 0.90: 15/64 (23.4375%)
- SSIM < 0.85: 4/64 (6.2500%)
- SSIM < 0.80: 1/64 (1.5625%)
- SSIM >= 0.90: 49/64 (76.5625%)
- SSIM >= 0.85: 60/64 (93.7500%)
- SSIM >= 0.80: 63/64 (98.4375%)

Worst case: `0013_v0` / `left_to_right` =
0.718375. The complete ordered tail is in
`ssim_direction_tail.csv`. Failures below 0.90 are concentrated in four base
images; base `0013` contains 6/8 such directions and the sole value below 0.80.

Temporal signed difference (`LATEACT - NEW`) has mean
0.001804, median 0.001691, range
[-0.007878, 0.018057]. All
64/64 pass the frozen >= -0.05 safeguard; the maximum observed
temporal deficit is 0.007878. Boundary-jump signed
difference has mean -0.001041, median
0.003666, and range [-0.661485,
0.291753]. Finite/nonconstant decode checks pass 64/64.

## Interpretation

The serving response and latency conclusions do not depend on treating 64
direction cells as independent: every base image has a positive effect and both
n=8 cluster-bootstrap intervals exclude zero. The future-SSIM result is not
supported only by its median: 60/64 directions are >=0.85, 63/64 are >=0.80,
and temporal/decode safeguards pass universally. Quality is nevertheless not
uniform: 15/64 are below 0.90 and one is 0.7184, with visible base-image
concentration. These unfavorable tails should be disclosed rather than converted
into a new gate.

## Recommended paper-level changes

1. Treat the 64 variant/direction cells as descriptive and use the eight base
   images for inference; report the n=8 cluster bootstrap and exact Wilcoxon.
2. Clarify whether 0.8159 denotes the fixed `direct_late` branch contrast. If
   the prose says all late arrivals, use the raw late-row n=8 estimate instead.
3. Add p05 and the exact SSIM tail counts, including the 0.7184 minimum and
   concentration in base image 0013; do not create or retroactively change a gate.
4. Retain the median >=0.90 frozen quality-gate statement, but do not imply that
   all 64 directions individually exceed 0.90.
