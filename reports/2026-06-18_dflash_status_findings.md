## DFlash Migration Status - 2026-06-18

### Scope

This note records the current state of the DFlash integration branch in the copied
experiment repo only:

- parent repo branch: `exp/dflash-integration-qw36-a2`
- `sglang` submodule branch: `exp/dflash-integration-qw36-a2`
- A2-only manual launches
- baseline service not touched

### Current status

The DFlash route is no longer blocked on startup alone.

Using a reduced-memory exploratory launch on A2:

- `--kt-num-gpu-experts 0`
- `--max-running-requests 1`
- `--max-total-tokens 32768`
- `--chunked-prefill-size 2048`
- `--speculative-dflash-draft-window-size 16`

the service now reaches:

- target model load
- target CUDA graph capture
- DFlash draft model load
- DFlash draft CUDA graph capture
- HTTP server startup
- `GET /model_info = 200`

and can answer requests instead of crashing during initialization.

### Most important differential finding

The highest-value comparison is now:

1. same copied repo
2. same copied env
3. same A2 GPU
4. same reduced-memory launch shape
5. only difference: `DFLASH` enabled vs disabled

#### DFlash enabled

`POST /generate`

Prompt:

`只回答最终结果：13乘以17等于多少？`

Observed response:

```json
{
  "text": "\n\n<think>\n\n</think>\n\n2023"
}
```

This is wrong.

Also observed:

- `spec_accept_rate = 0.0`
- `spec_accept_length = 1.1111111111111112`

For a JSON-only prompt:

`请直接输出一个合法JSON：{\"result\":221}`

the DFlash route produced a long reasoning-style continuation instead of valid JSON.

#### Same branch, DFlash disabled

Using the same copied branch and same reduced-memory launch, but without the
`--speculative-*` DFlash arguments:

`POST /generate`

Prompt:

`只回答最终结果：13乘以17等于多少？`

Observed response:

```json
{
  "text": "\n\n<think>\n\n</think>\n\n221"
}
```

This is semantically correct.

### Interpretation

This isolates the current main blocker:

- the migration has progressed past service startup
- the copied branch itself can still produce correct raw generation without DFlash
- enabling the DFlash path breaks raw generation semantics

So the active blocker is now **DFlash/speculative correctness**, not:

- baseline contamination
- missing model files
- startup wiring
- or generic copied-environment corruption

### Secondary observations

- `POST /v1/chat/completions` with `max_tokens=16` still tends to stop inside
  reasoning content on both paths, so raw `/generate` is currently the better
  correctness probe for this branch.
- The DFlash-enabled path remains low-value from a decode perspective so far,
  because acceptance is effectively collapsing near 1 token.

### Local artifacts

- DFlash startup log:
  - `reports/dflash_launch_20260618_023420.log`
- same-branch no-spec control log:
  - `reports/dflash_control_nospec_20260618.log`

### Current conclusion

The migration is now at a meaningful checkpoint:

- DFlash can be integrated deeply enough to start serving on the copied branch
- but it is not yet acceptable for deployment because output correctness is damaged

The next debugging target should be DFlash verify / accepted-token correctness,
not more startup plumbing.
