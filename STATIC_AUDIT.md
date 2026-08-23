# Phase -1 zero-generation static audit

Status: **PASS**. The late-action intervention is implementable without editing
the foundation model or restarting the current latent.

Evidence refers to the official Matrix-Game checkout at commit
`71c3cd7f741311f8100f6cf9cde942b6c1378d11`.

1. **Sampler schedule — PASS.**
   `configs/inference_yaml/inference_universal.yaml:1-6` declares raw distilled
   steps `[1000, 666, 333]` with warping enabled. The scheduler transform in
   `pipeline/causal_inference.py:150-154` yields actual timesteps
   `[1000.0, 908.8427124023438, 713.9794311523438]`: exactly three denoiser
   evaluations per block. `causal_inference.py:335-347` then performs one
   separate context-cache write at `context_noise=0`; it is not counted as an
   NFE because its prediction is not sampled into the block output.

2. **Action acquisition — PASS.**
   Interactive universal actions are read at
   `pipeline/causal_inference.py:14-44`; J/L map to horizontal mouse values
   `-0.1/+0.1`. Offline inference obtains benchmark action tensors at
   `inference.py:120-131`. Project-side tensors reproduce the same two values
   while keeping keyboard inputs zero.

3. **Every-evaluation conditioning — PASS.**
   Each official denoiser call independently invokes `cond_current` at
   `causal_inference.py:293-330`. That function slices `mouse_cond` and
   `keyboard_cond` through the current block at lines 110-127. The model passes
   both tensors through every transformer block to the action module at
   `wan/modules/causal_model.py:273-288`.

4. **Replacement between evaluations — PASS.**
   `conditional_dict` is an ordinary argument to each generator call; no action
   is captured outside the call. Selecting an old or new condition immediately
   before each of the three calls changes action input without resetting
   `noisy_input`. The official implicit re-noising at
   `causal_inference.py:311-318` is replaced only in the project wrapper by two
   explicit, pre-hashed tensors.

5. **Repeated current-block action-cache behavior — PASS.**
   Mouse cache positions are computed from `start_frame` at
   `action_module.py:277-304`; keyboard uses the same rule at lines 405-431.
   On the first current-block evaluation, `current_end > global_end_index`, so
   the three new positions append. On later evaluations of the same
   `current_start`, the delta is zero and the same three positions are
   overwritten. This is verified dynamically by per-step cache indices and
   hashes in the smoke.

6. **Historical/current isolation — PASS, subject to dynamic assertion.**
   The experiment uses one three-latent prefix and one three-latent future
   block. Action-cache capacity is six latent positions
   (`causal_inference.py:713-735`, model `local_attn_size=6`), so no eviction is
   triggered. The prefix slice `[0:3]` must remain bit-identical while `[3:6]`
   is overwritten. The runner fails closed if any layer violates this.

7. **Other branch state — PASS, subject to dynamic assertion.**
   Visual self-attention, mouse, keyboard, and cross-attention caches are cloned
   from one prefix snapshot before every branch. The visual prefix slice is
   asserted unchanged during the current block. Cross-attention K/V is computed
   once and reused (`wan/modules/model.py:241-250`) and must remain hash-identical.
   Fixed image conditioning, model precision, sampler, initial noise, and both
   re-noising tensors are hashed. Only the current mouse action values differ.

The upstream checkout already contains a pre-existing, auditable visual-cache
storage hook from an earlier project. LateAct does not modify it and uses only
its ordinary BF16 path.

