# LateAct showcase package

This directory is a self-contained, web-ready visual package built only from frozen LateAct outputs. It makes the supported Matrix-Game 2.0 mouse-yaw story legible in 30–60 seconds: late action arrival, denoising-time commitment, direct-bind failure, minimal rollback, and lower latency than full restart. No inference or scientific recomputation was used to create it.

## Recommended page sequence

1. **Hero:** `og_cover.png`, title, hook, and the three values in `headline_metrics.json`
2. **Demo:** `lateact_demo.mp4` with `demo_thumbnail.png` as its poster
3. **Mechanism:** `lateact_mechanism.svg` (PNG fallback included)
4. **Results:** `commitment_curve.svg`, then `rollback_state.png`
5. **Limitations:** the scope paragraph from `PROJECT_PAGE_COPY.md`

## Asset manifest

| asset | purpose | original source | frozen-result support | placement |
|---|---|---|---|---|
| `og_cover.png` | 1200×630 LinkedIn/Open Graph card using real model frames and a compact rollback cue | Gate 1 case 0005, right-to-left | Directly supported | Hero / social card |
| `lateact_demo.mp4` | Synchronized four-way H.264 comparison: direct, LateAct, full restart, NEW oracle | Gate 1 case 0005, right-to-left | Directly supported | Demo |
| `demo_thumbnail.png` | Standalone poster/still for the demo | Same Gate 1 case | Directly supported | Demo poster / fallback |
| `lateact_mechanism.svg` | Vector explanation of direct binding, minimal rollback, and full restart | Frozen LateAct mechanism | Directly supported conceptual rendering | Mechanism |
| `lateact_mechanism.png` | Raster fallback for the mechanism figure | Same as SVG | Directly supported conceptual rendering | Mechanism fallback |
| `commitment_curve.svg` | Vector curve with all 16 validated mouse-yaw traces and frozen median annotations | Gate 0 `gate0_summary.json` | Directly supported | Results |
| `commitment_curve.png` | Raster fallback for the commitment curve | Same as SVG | Directly supported | Results fallback |
| `rollback_state.png` | Initial exact, final exact, and logical-minimum checkpoint comparison | Gate 1 + Gate 2 primary summaries | Directly supported | Results / systems |
| `headline_metrics.json` | Exact/display headline values plus source paths and JSON fields | Gate 1 + Gate 2 primary summaries/reports | Directly supported | Result cards / page data |
| `PROJECT_PAGE_COPY.md` | Ready-to-adapt technical landing-page copy and conservative scope language | Frozen reports and summaries | Directly supported | All sections |
| `ASSET_AUDIT.md` | Inventory of existing videos, controls, montages, data, and scripts | Full repository/artifact audit | Provenance document | Maintainer reference |
| `POLISH_NOTES.md` | Figure-level revision log and explicit scientific-invariant check | Current showcase assets and frozen values | Presentation-only provenance | Maintainer reference |
| `canonical_example.json` | Machine-readable selection rule, exact case metrics, and clip mapping | Gate 1 per-pair record | Directly supported | Maintainer reference |
| `source_clips/canonical_{old,new,direct,lateact,restart}.mp4` | Local copies of all five canonical conditions | Gate 1 case 0005, right-to-left | Direct frozen outputs | Provenance / future edits |
| `frozen_sources/gate0_summary.json` | Local data snapshot for commitment curve | Gate 0 summary | Direct frozen output | Provenance |
| `frozen_sources/gate1_summary.json` | Local data snapshot for canonical case and Gate 1 response | Gate 1 summary | Direct frozen output | Provenance |
| `frozen_sources/gate2_mouse_summary.json` | Local data snapshot for latency/checkpoint cards | Gate 2 primary summary | Direct frozen output | Provenance |
| `build_showcase.py` | Deterministic lightweight composition script; never imports the model runtime | Inputs above | Derivative-only | Maintainer reference |

## Canonical case

Validated Gate 1 scene `0005`, `right_to_left`, was chosen using quantitative representativeness before visual polish: direct response `0.1915` versus cohort median `0.2182`, LateAct response `0.9966` versus cohort median `0.9971`, future SSIM `0.9210`, and passing decode/state audits. It cleanly shows direct late binding retaining the OLD yaw while LateAct, full restart, and the NEW oracle converge visually. See `canonical_example.json` for unrounded values and exact provenance.

## Build

From the repository root, using the existing lightweight environment:

```bash
python3 showcase/build_showcase.py
```

Set `LATEACT_ARTIFACT_ROOT` to the frozen artifact directory before rebuilding. The script reads videos/JSON, copies small provenance inputs, and composes figures/video; it does not load Matrix-Game or use a GPU. Rebuilding requires the lightweight Pillow, OpenCV, NumPy, and imageio dependencies.

## Remaining material

No missing scientific or visual asset blocks a polished GitHub Pages landing page. Publication still requires ordinary site assembly and final public URLs for the repository and reports. Optional non-scientific refinements would be WebVTT captions/alt text in the page markup and a compressed mobile video rendition; the supplied MP4, poster, SVGs, source clips, copy, and provenance are sufficient for the page itself.

## Claim boundary

The efficacy claim is Matrix-Game 2.0 mouse-yaw only. Keyboard shares the measured boundary but fails the frozen future-SSIM trajectory-fidelity requirement. minWM is exploratory commitment-like evidence whose formal replication gate failed; it is not a successful cross-model replication.
