# Vendored BranchSafe cache dependency

LateAct's Matrix-Game execution used a pre-existing cache-storage hook from the
local Branch-Safe Runtime-State Compression project. That project has no remote,
so the minimum dependency required by the frozen Matrix-Game patch is vendored
here.

Provenance:

- source repository: local `branch-safe-kv`
- source commit: `6757e11` (`record Gate 0 NO-GO`)
- `branchsafe/cache_ops.py` source SHA256:
  `af30ac8f5d58c1b310cfce74e1a4d381dcfdfe7a0acfbf08db9920ec929f5e12`
- Matrix-Game base commit:
  `71c3cd7f741311f8100f6cf9cde942b6c1378d11`
- patch: `../../patches/matrix_game_cache_hook.patch`

The LateAct experiments use the hook's ordinary BF16 cache behavior. The file
also contains the original INT8/INT4 helpers because preserving the source file
byte-for-byte is safer than extracting a rewritten subset.

Install into a recovery environment with:

```bash
python -m pip install -e recovery/vendor/branchsafe
```

Run the vendored unit tests with:

```bash
python -m pytest recovery/vendor/branchsafe/tests
```
