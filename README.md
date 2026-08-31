# LateAct: Commitment-Aware Asynchronous Control for Interactive Video World Models

LateAct restores the latest pre-commitment denoising state when an action arrives late, then recomputes only the affected suffix instead of restarting the full block.

[Project page](./index.html) · [Technical report](./GATE2_REPORT.md) · [Demo](./showcase/lateact_demo.mp4)

![LateAct comparison: direct late binding retains the old yaw while LateAct tracks the new-action oracle.](./showcase/og_cover.png)

## Validated scope

The primary validated result is **Matrix-Game 2.0 mouse-yaw control**. The project studies asynchronous action arrival during iterative video denoising: when direct late binding is no longer effective, can an exact minimal rollback recover the newly requested action while preserving reusable computation?

## Main results

- **Commitment boundary:** across 16 validated mouse-yaw scene-direction pairs, median normalized NEW-action response falls from `0.997` after one OLD-action NFE to `0.218` after two.
- **Minimal rollback:** Gate 1 raises median late-action response from `0.218` for direct late binding to `0.997`; all 16/16 validated pairs improve and remain within 0.10 of full restart.
- **Fresh asynchronous serving:** Gate 2 evaluates 32 fresh contexts. All 676/676 late arrivals remain within 0.10 of full restart, while all 348/348 early arrivals use zero rollback.
- **Latency and state:** LateAct measures `11.93%` lower mean post-arrival latency than full restart with an exact `339,360`-byte rollback checkpoint.

Exact display values and source fields are recorded in [`showcase/headline_metrics.json`](./showcase/headline_metrics.json). Frozen reports are available in [`GATE0_REPORT.md`](./GATE0_REPORT.md), [`GATE1_REPORT.md`](./GATE1_REPORT.md), and [`GATE2_REPORT.md`](./GATE2_REPORT.md).

## Code and reproduction notes

- Experiment runners and analyses are under [`scripts/`](./scripts/); reusable runtime and rollback code is under [`src/lateact/`](./src/lateact/).
- Frozen protocols are under [`config/`](./config/). Copy [`config/storage.env.example`](./config/storage.env.example) to the ignored `config/storage.env` and set paths for your machine.
- Matrix-Game/minWM repositories, checkpoints, datasets, caches, and full generated artifacts are intentionally excluded. See [`STATIC_AUDIT.md`](./STATIC_AUDIT.md) and the gate reports for audited upstream versions and protocol details.
- The static project page is [`index.html`](./index.html) with no build step; preview from the repository root with `python3 -m http.server 8000`.

## Limitations

- The strongest efficacy evidence is Matrix-Game 2.0 mouse-yaw control.
- Keyboard exhibits the same measured commitment boundary, but fails the frozen future-SSIM trajectory-fidelity requirement.
- The original minWM native `a/d` study remains a formal 6/16 failure of the
  frozen oracle-separation gate. Exactly one subsequently frozen `j/l` yaw
  confirmation passed all 16/16 commitment gates and independently replicates
  the denoising-time commitment mechanism only.
- minWM did not test LateAct rollback or latency. Full LateAct efficacy remains
  validated only for Matrix-Game 2.0 mouse-yaw, and the method is not
  established as universally action-general.

## Disaster recovery

The private recovery snapshot is documented in
[`recovery/RECOVERY.md`](./recovery/RECOVERY.md). Frozen numerical records are
archived under [`results/frozen/`](./results/frozen/); licensed upstream model
weights and bulk generated videos/latents are intentionally excluded.
