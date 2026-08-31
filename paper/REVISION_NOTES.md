# LateAct paper revision notes

## 1. Novelty defense

The paper now explicitly accepts the known premise that conditioning influence
can vary over diffusion time. It positions LateAct around a different chain:
controlled asynchronous action substitution, an operational recoverability
boundary, and exact suffix rollback. Claims of a universally sharp or
infinitely discrete boundary were removed, and the causal language is scoped
to a controlled intervention on the model rollout.

## 2. Related work

Related work now covers timestep-varying conditioning and editing
(SDEdit, Prompt-to-Prompt, and CADS) and cached diffusion computation
(DeepCache), in addition to interactive world models and few-step video
generation. Each added citation was verified against its primary
OpenReview/CVF publication record. The text distinguishes semantic editing,
conditioning schedules, and approximate feature reuse from LateAct's
asynchronous action-recoverability measurement and exact checkpoint/replay
policy.

## 3. Statistical unit correction

The main serving inference now uses eight independent base-image clusters,
not 64 derived variant/direction cells. The hierarchy is stated explicitly:
8 bases, 4 deterministic variants per base, 2 directions per variant, and
16 arrivals per direction. The 64 direction cells remain descriptive quality
units.

## 4. Response estimand

The primary response statistic is the actual asynchronous late-arrival
estimand over all 676 frozen late rows: mean LateAct minus Direct `0.909136`,
median `0.930385`, base-cluster bootstrap 95% CI
`[0.859769, 0.954093]`, with 8/8 positive bases. The former `0.815919` value is
retained only in the appendix and explicitly labeled as a distinct descriptive
fixed-branch direction-cell contrast.

Latency inference likewise uses eight base clusters: mean Full-Restart minus
LateAct `0.311789 s`, median `0.313654 s`, 95% CI
`[0.305519, 0.317523]`, and 8/8 positive bases. The frozen `11.93%` mean
late-arrival reduction is preserved.

## 5. SSIM-tail disclosure

The main text reports mouse-yaw median future-SSIM `0.9223`, p05 `0.8397`,
`60/64 >= 0.85`, and universal temporal/decode safeguards. The appendix gives
the mean, full requested threshold counts, failure concentration, and the
unfavorable `0.718375` minimum. The `0.90` safeguard is explicitly described
as median-level rather than per-example, and quality is characterized as
broadly stable but not uniform.

## 6. minWM follow-up wording

The chronology now states that the original lateral `a/d` experiment formally
failed at 6/16 because of inadequate oracle separation. Exactly one bounded
`j/l` yaw follow-up was frozen at commit `f1e2bbc` after that identifiability
audit and before confirmatory outputs were generated, using fresh consecutive
prompts 8--15, seeds 41008--41015, and unchanged evaluator, thresholds, and
gate. It is called a pre-specified frozen follow-up, not an external
preregistration. minWM supports commitment-mechanism replication only; no
rollback or latency transfer is claimed.

## 7. Threshold rationale

The paper now explains the protected failure mode for the 8-pixel oracle gap,
10x repeatability margin, 20-inlier minimum, 0.10 near-restart tolerance,
median SSIM `0.90` safeguard, and 0.15 adjacent-drop criterion. It does not
claim that these numerical values are uniquely correct.

## 8. Algorithm

A compact architecture-independent algorithm now separates calibration from
serving: sweep coupled action switches, identify the latest safe state, bind
early arrivals directly, and restore/replay only the necessary suffix for late
arrivals. A separate dependency-audit step derives the exact checkpoint
payload. The text explicitly states that Matrix-Game's 339,360-byte payload
does not automatically transfer to another architecture.

## 9. Figure and table changes

The conceptual rollback and cross-model commitment figures remain in the main
paper at smaller sizes. The favorable qualitative example was moved to the
appendix and labeled as illustrative rather than distributional evidence. No
new model outputs or model-derived figures were produced. The serving-table
caption now clarifies that Next-Block response refers to the current block,
whereas its latency ends when the new action first affects following-block
pixels.

## 10. Claims intentionally not strengthened

The revision does not claim minWM rollback efficacy, minWM latency savings,
cross-model validation of the LateAct policy, action-general trajectory
fidelity, universal commitment, exact trajectory identity, or physical causal
correctness. Keyboard retains the shared-boundary and recovered-direction
result together with the failed `0.6726 < 0.90` trajectory-fidelity gate.

## 11. Remaining reviewer vulnerabilities

Full rollback efficacy is still established on one model and one action
family. The statistical inference has only eight independent base images.
Optical-flow response metrics are action-specific, mouse-yaw quality has a
nonuniform tail, and keyboard shows that recovered direction need not recover
the intended trajectory. Thresholds are defensible safeguards rather than
universally validated operating points. Timing is implementation- and
RTX-3090-specific, and no deployment scheduler or physical task is evaluated.

The compiled manuscript remains anonymous and keeps the main paper to six
pages, with references beginning on page 7.
