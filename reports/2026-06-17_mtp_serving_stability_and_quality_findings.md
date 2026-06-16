# 2026-06-17 KT-MTP Serving Stability And Quality Findings

## Scope

This note captures the latest authoritative runtime findings for the copied KT-MTP experiment only:

- repo: `/data/nvme0/ktransformers_qw36_mtp_exp`
- env: `/data/nvme0/kt_a2_qw36_mtp_exp`
- GPU: `GPU-8625457e-0c7b-dd40-5d76-1f5845d811ae`

Baseline systemd and baseline editable environment were not modified or restarted.

## What Changed In This Phase

This phase was focused on converting the earlier "first real request crashes" state into a stable serving state, then checking whether output quality and decode performance were actually acceptable.

Key code progress committed during this phase:

- `6c56002 / 9f35b37c9`
  materialize speculative outputs on CPU in non-overlap flow
- `eb9ec26 / 3a32545b7`
  add host-side accepted-token snapshot path for spec-v2 outputs
- `821a0f6 / 2ab59b4a7`
  clone speculative verify outputs before reuse
- `657e0f0 / 2da27d210`
  derive accept lengths from seq-len deltas
- `33b760c / 5699a7033`
  disable target verify cuda graph for KT-MTP
- `806ee8a / 1c6eac473`
  gather accepted spec-v2 tokens by `accept_index` instead of contiguous slicing

## Stable Result Achieved

The most important runtime improvement is that KT-MTP serving no longer dies on the first real request.

Authoritative evidence:

- `POST /v1/chat/completions -> 200 OK`
- `/health -> 200`
- repeated request handling continued after startup instead of crashing immediately

This means the work has crossed an important threshold:

- the service is now stable enough to serve multiple OpenAI-compatible requests
- quality failures are now observable as model/runtime behavior, not just hidden behind early crashes

## Quality Result

Quality is currently not acceptable.

Direct manual sample:

- prompt:
  - `只回答最终结果：13乘以17等于多少？`
- observed output:
  - repeated `</think>` tokens instead of `221`

Structured quality validation was run with:

- report:
  - `/data/nvme0/ktransformers_qw36_mtp_exp/reports/mtp_quality_run_20260617_0231.json`

Result:

- `all_passed = false`
- all 5 cases failed

Failure shape:

- empty visible output on some cases
- malformed reasoning text such as repeated `Here's user...`
- visible output polluted by repeated `</think>`
- JSON formatting failure
- simple exact-answer failure

This means requirement 2 from the active objective is not satisfied.

## Performance Result

The current KT-MTP path is also not meeting the performance goal yet.

Observed server info near the end of the successful stable run:

- `avg_spec_accept_length = 1.0`
- `last_gen_throughput ~= 5.32 tok/s`

Interpretation:

- speculative decoding is effectively accepting only the minimum path
- the path is not delivering meaningful MTP decode gain
- the measured decode behavior is well below the baseline reference of about `16 tok/s`

So requirement "MTP enabled should improve decode speed" is also not satisfied.

## Important Inference

The latest fixes solved serving stability, but they did not solve semantic correctness.

Given:

- target verify cuda-graph replay was a real crash source and is now bypassed
- serving is stable enough to answer multiple requests
- accepted-length statistics remain pinned at `1.0`
- quality remains severely corrupted

the current blocker has shifted from "runtime crashes" to "speculative correctness".

The most likely remaining problem class is not general server plumbing anymore. It is deeper inside one or both of:

- spec-v2 accepted-token semantics on the KT-MTP path
- reasoning/thinking token handling under speculative verification and output assembly

## Objective Status Against Requirements

### 1. Prevent MTP-related draft pieces from living permanently on GPU while allowing some experts on CPU

Partial only.

Confirmed:

- KT-backed draft MTP experts are active
- `mtp.layers.0` BF16 draft expert path works
- CPU/GPU mixed expert placement is real

Not finished:

- no implemented stable residency policy yet for draft `embed`
- no implemented stable residency policy yet for draft `lm_head`
- no implemented stable residency policy yet for draft non-expert layers

### 2. Validate output quality

Completed as a validation activity, but failed as a quality outcome.

Confirmed:

- service can now be quality-tested
- quality report exists
- report proves output quality is currently unacceptable

## Recommended Next Stage

The next phase should treat the problem as speculative correctness repair, not startup debugging.

Most direct continuation:

1. audit how accepted tokens are assembled and surfaced in spec-v2 KT-MTP
2. audit interaction between speculative outputs and `qwen3-thinking` reasoning parser behavior
3. only after outputs become semantically correct, revisit decode speed and draft non-expert residency

## Current Runtime Cleanup Status

At the end of this phase:

- manual experiment service was stopped
- A2 memory returned to idle baseline state
