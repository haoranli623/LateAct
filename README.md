# LateAct

**Commitment-Aware Asynchronous Control for Interactive Video World Models**

LateAct studies asynchronous action arrival during iterative video generation and uses commitment-aware minimal rollback to recover late control without unnecessarily restarting the full block.

[Project Page](https://haoranli623.github.io/LateAct/) · [Demo](https://haoranli623.github.io/LateAct/#demo)

## Validated scope

The primary validated result is Matrix-Game 2.0 mouse-yaw control.

- Median late-action response: **0.218 → 0.997**
- Mean latency vs. full restart: **11.93% lower**
- Exact rollback state: **339 KB**

The strongest efficacy evidence is currently limited to the validated Matrix-Game 2.0 mouse-yaw setting. Broader action-family and cross-model generality are not claimed.

Code and the complete technical report are not currently part of this public project-page release.
