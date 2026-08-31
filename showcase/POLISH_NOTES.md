# Showcase polish notes

Polish pass: 2026-08-27. This was a presentation-only revision. No experiment was rerun, no frozen artifact was modified, and no scientific value, conclusion, or supported scope changed.

| issue fixed | files changed | resolution | wording changed? |
|---|---|---|---|
| Commitment-chart text collision and awkward y-axis treatment | `commitment_curve.png`, `commitment_curve.svg`, `build_showcase.py` | Increased plot-top clearance, established a title/subtitle/axis hierarchy, added a conventional vertical y-axis label, and retained full-range axes. | Subtitle was simplified to describe Gate 0, sample count, and thin per-pair traces; semantics are unchanged. |
| Commitment transition was visually crowded | `commitment_curve.png`, `commitment_curve.svg`, `build_showcase.py` | Preserved all 16 per-pair curves and the aggregate curve, moved the `0.997` and `0.218` pills away from linework, and placed the paired median drop in a separate low-contrast callout. Endpoint labels still identify `x=0` as NEW oracle and `x=3` as OLD oracle. | No quantitative wording changed. |
| Direct Late Bind thumbnail label was cramped | `demo_thumbnail.png`, `build_showcase.py` | Increased safe text padding, wrapped the long label, regularized card alignment, and kept the 2×2 four-condition layout. | No condition name or value changed. |
| Representative-case metrics could be mistaken for aggregate results | `demo_thumbnail.png`, `build_showcase.py` | Added a visible `Representative validated Gate 1 case · case-specific responses` line and a `case-specific` tag under each displayed response. The footer now identifies Gate 1 scene 0005, right-to-left, and matched generation controls. | Clarifying wording added only; case values remain `0.191`, `0.997`, and `1.000`. |
| Mechanism title was less natural | `lateact_mechanism.png`, `lateact_mechanism.svg`, `build_showcase.py` | Changed the title from `LateAct crosses only the commitment boundary` to `LateAct rolls back only across the commitment boundary`. | Yes, for clarity only; mechanism and claim are unchanged. |
| Full-restart return arc looked improvised | `lateact_mechanism.png`, `lateact_mechanism.svg`, `build_showcase.py` | Replaced the rough lower arc with a deliberate purple `BLOCK START` restart node feeding the full NFE1→NFE2→NFE3 chain. | `BLOCK START` was added as a schematic label. |
| Arrow and condition styling was inconsistent | `lateact_mechanism.png`, `lateact_mechanism.svg`, `build_showcase.py` | Standardized arrow widths and endpoint colors; Direct remains orange/red, LateAct is consistently teal, Full Restart remains purple, and the amber commitment boundary remains distinct from the coral arrival marker. | No scientific wording changed. |
| OG-cover integration check | `og_cover.png`, `build_showcase.py` | Verified 1200×630 dimensions, safe outer margins, alignment, readable typography, and matching Direct/LateAct color semantics. No visual change was necessary. | No. |

## Scientific invariants confirmed

- Aggregate Gate 1 headline response remains `0.218 → 0.997`.
- The representative Gate 1 case remains `0.191 → 0.997`, with Full Restart at `1.000`.
- The commitment plot still uses all 16 validated scene-direction pairs and retains the paired median drop `0.774`.
- `x=0` remains the NEW oracle endpoint; `x=3` remains the OLD oracle endpoint.
- The supported efficacy scope remains Matrix-Game 2.0 mouse-yaw. No action-general or cross-model claim was added.

## Page-integration refinements

- The commitment-curve subtitle now explicitly maps thin lines to the 16 validated pairs and the bold blue line to the median trend.
- The mechanism title now reads `LateAct rolls back to the latest pre-commitment state`; the underlying diagram and takeaway are unchanged.
