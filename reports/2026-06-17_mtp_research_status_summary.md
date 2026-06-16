# 2026-06-17 Qwen3.6 A2 KT-MTP Research Status

## Scope

This report summarizes the current MTP exploration status for the copied experiment workspace only:

- repo: `/data/nvme0/ktransformers_qw36_mtp_exp`
- env: `/data/nvme0/kt_a2_qw36_mtp_exp`
- model: `/home/lsp/data/models/gptq/qwen3.6-35b-a3b`
- GPU: `GPU-8625457e-0c7b-dd40-5d76-1f5845d811ae`

The baseline editable environment and baseline systemd service were kept untouched after VRAM release. No baseline restart was performed.

## Executive Summary

The MTP effort is no longer at the "feature switch" stage. It has progressed into a partial runtime integration:

- draft MTP expert routing through KT BF16 is real and verified in runtime logs
- speculative draft worker wiring has been extended across the relevant SGLang worker paths
- the service can initialize much further than at the beginning, including target load, draft load, CUDA graph capture, and `GET /model_info -> 200`

However, the end state is still blocked:

- first real speculative request is not yet proven stable
- output quality validation has not completed
- decode speedup over the `~16 tok/s` baseline has not been measured
- draft non-expert residency policy is not solved yet, so "MTP-related lm_head/layers not permanently on GPU" is still incomplete

## What Has Been Proven

### 1. KT-backed MTP draft experts are actually active

Runtime logs already showed all of the following:

- `KT MTP draft mode enabled ... method=BF16 ... prefix=mtp.layers`
- `Using KT draft-MTP configuration: layer_idx=0 gpu_experts=48/256 method=BF16 prefix=mtp.layers`
- `Created AMX_BF16_MOE_TP 0 at numa 0`

This is the strongest confirmed result so far. It proves:

- `mtp.layers.0.*` weights can be loaded through KT expert infrastructure
- the draft path is not staying on the default GPU-only MoE implementation
- partial GPU/CPU expert placement for the MTP draft layer is feasible on A2

### 2. The copied SGLang stack has been pushed into real draft runtime

The experiment no longer stops at trivial boot errors. After a sequence of targeted fixes, the stack has already been driven through:

- target model weight load
- target CUDA graph capture
- draft model weight load
- draft CUDA graph capture
- uvicorn startup
- `GET /model_info -> 200`

### 3. The draft execution chain now has a coherent KT-MTP control plane

The copied `third_party/sglang` now contains a dedicated draft-KT control surface, including:

- `kt_mtp_enable`
- `kt_mtp_weight_path`
- `kt_mtp_method`
- `kt_mtp_num_gpu_experts`
- `kt_mtp_gpu_experts_ratio`
- `kt_mtp_max_deferred_experts_per_token`

And the draft path is explicitly rewritten to use:

- `kt_weight_prefix = "mtp.layers"`
- `kt_num_layers_override = 1`

That control plane has also been propagated across the main speculative worker variants, instead of staying as a one-off patch.

## Main Repairs Already Implemented

The following categories were already repaired in the copied submodule:

1. Missing draft backend warmup callbacks
   - fixed for both FlashInfer and Triton multi-step draft backends
2. Qwen3.5/3.6 MTP model-path incompatibilities
   - accepted legal `input_embeds`
   - relaxed an over-strict multimodal embed guard
3. Draft MTP layer loading through KT
   - added BF16 draft-MTP prefix loading for `mtp.layers.0.*`
4. Draft runtime path stabilization
   - forced a safer draft attention backend for KT-MTP
   - added draft KV cache dtype fallback when `triton + fp8` was unsafe on A2
5. `DRAFT_EXTEND_V2` compatibility work
   - mrope handling
   - hybrid-attention extend handling
   - Mamba metadata guards
   - state carryover in non-overlap spec-v2 flow
6. Non-overlap speculative path alignment
   - kept KT-MTP on `EAGLEWorkerV2`
   - made scheduler/batch/spec-output handling consistent with spec-v2 semantics even with overlap disabled

## Current Hard Blockers

The remaining blockers are now narrower, but they are still real blockers.

### 1. First real speculative request is still not proven stable

The current stack can start, but the experiment has not yet demonstrated a clean, repeatable first OpenAI-compatible generation request followed by stable service health.

The latest debugging direction narrowed the problem to the first real request path after prefill, not to model boot or parameter wiring.

### 2. Non-overlap + spec-v2 + draft runtime still needs final validation

Because KT-MTP draft mode was forced onto a more stable non-overlap scheduler path, multiple scheduler/batch/output semantics had to be realigned around `EAGLEWorkerV2`.

Those compatibility patches are now in place, but they still need a fresh end-to-end rerun to confirm the chain is finally coherent under a real request.

### 3. Draft non-expert GPU residency is still unsolved

The original optimization target was broader than "load MTP experts through KT". It also required preventing MTP-related `lm_head`, draft layer, and similar non-expert pieces from becoming long-lived GPU residents.

That part is not finished. The current work mainly proves KT expert routing and speculative runtime integration. A separate residency/staging design is still required for:

- draft `embed`
- draft `lm_head`
- draft non-expert layer weights

### 4. Quality and performance objectives are still pending

The following have not been completed yet:

- OpenAI API quality validation on meaningful prompts
- proof that output quality is not damaged
- decode throughput measurement with MTP enabled
- confirmation that MTP beats the baseline `~16 tok/s`

## Most Relevant Existing Artifacts

Earlier reports and tools already present in the experiment repo:

- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/2026-06-15_mtp_exploration_block_summary.md`
- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/2026-06-15_mtp_kt_full_development_plan.md`
- `/data/nvme0/ktransformers_qw36_mtp_exp/tools/mtp_quality_cases.json`
- `/data/nvme0/ktransformers_qw36_mtp_exp/tools/validate_serving_quality.py`

Recent manual runtime logs used during this phase:

- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/sglang_qw36_mtp_manual_20260617_012125.log`
- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/sglang_qw36_mtp_manual_20260617_012536.log`
- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/sglang_qw36_mtp_manual_20260617_013611.log`
- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/sglang_qw36_mtp_manual_20260617_014119.log`
- `/data/nvme0/ktransformers_qw36_mtp_exp/reports/sglang_qw36_mtp_manual_20260617_014530.log`

## Suggested Next-Stage Goals

If a new goal is created, it should be explicit about which unfinished layer is being targeted.

### Option A: Finish runtime stabilization first

Goal shape:

- make `kt_mtp_enable + non-overlap + EAGLEWorkerV2` survive the first real request
- require `/health = 200` after generation
- do not start performance work before this is true

This is the most direct continuation of the current line of work.

### Option B: Solve draft non-expert residency separately

Goal shape:

- design and implement a CPU-resident or staged strategy for draft `embed`, `lm_head`, and non-expert weights
- keep the KT draft expert path intact

This is necessary if the next phase prioritizes VRAM discipline rather than immediate serving stability.

### Option C: Only proceed if quality validation becomes possible

Goal shape:

- require a stable local OpenAI-compatible request path first
- then run `tools/validate_serving_quality.py`
- only after quality passes, run decode-speed measurement

This is the most conservative gate if correctness must dominate performance work.

## Bottom Line

The exploration has already produced one meaningful result: KT-backed BF16 MTP draft experts are working on the copied Qwen3.6 stack and can be partially placed off-GPU. But the project is not yet in a "measure speedup" state.

The next goal should treat the remaining work as real runtime integration and stabilization, not as a small switch-flip or minor benchmark exercise.
