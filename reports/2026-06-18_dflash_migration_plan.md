# 2026-06-18 DFlash Migration Plan For Current KT Deployment

## Goal

Evaluate whether the DFlash drafter model `z-lab/Qwen3.6-35B-A3B-DFlash` can be integrated into the current copied deployment stack, without touching the baseline environment, and with clear isolation from the current KT-MTP line of work.

This plan is for:

- repo: `/data/nvme0/ktransformers_qw36_mtp_exp`
- env: `/data/nvme0/kt_a2_qw36_mtp_exp`
- target model today: `/home/lsp/data/models/gptq/qwen3.6-35b-a3b`
- GPU constraint: A2 only

## Bottom-Line Assessment

DFlash is not a small patch on top of the current KT-MTP branch.

It is a separate speculative decoding stack that introduces:

- a new speculative algorithm (`DFLASH`)
- new worker/data-structure code paths
- new server args and validation rules
- target-model hidden-state capture requirements
- new cuda-graph and KV-memory planning behavior

So the migration should be treated as a parallel integration track, not as a continuation of the current EAGLE/KT-MTP repair branch.

## Source To Migrate

Reference implementation inspected from:

- upstream SGLang PR checkout:
  - `/data/nvme0/sglang_pr20547_inspect`
- PR head:
  - `e67a0d488d905661e621342912874bc7893f1d94`

Current local copied SGLang base:

- `/data/nvme0/ktransformers_qw36_mtp_exp/third_party/sglang`
- current base commit after baseline reset:
  - `ef0f44982fb9cb6359b199de7624e68fcd18f0df`

## Migration Strategy

### Principle 1: Do not mix DFlash with current KT-MTP repair branch

Create a dedicated branch for DFlash integration.

Recommended branch names:

- parent repo: `exp/dflash-integration-qw36-a2`
- submodule: `exp/dflash-integration-qw36-a2`

Reason:

- current KT-MTP branch already contains many speculative-specific hotfixes
- DFlash changes many of the same framework files
- mixing them will make regression ownership unclear

### Principle 2: Validate raw generation correctness before KT-specific optimization

Do not start from "make DFlash use KT expert routing".

First milestone must be:

- target + DFlash drafter can start
- `/generate` raw output is semantically correct
- `/v1/chat/completions` output is acceptable

Only after that should KT-specific acceleration or placement work resume.

### Principle 3: Keep baseline launch arguments unchanged unless adding DFlash-specific args

The deployment should preserve the current target serving shape and only add:

- `--speculative-algorithm DFLASH`
- `--speculative-draft-model-path ...`
- `--speculative-num-draft-tokens ...`
- optionally `--speculative-dflash-draft-window-size ...`

Do not mutate unrelated baseline arguments during the first migration pass.

## Required Code Areas

The PR is not localized. These are the minimum areas that must be ported together.

### 1. Speculative framework layer

Must migrate:

- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_info_v2.py`
- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_worker_v2.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/draft_utils.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`

Reason:

- this is where the `DFLASH` algorithm is introduced
- worker selection, draft/verify state, and block-level acceptance live here

### 2. Server arg and validation layer

Must migrate:

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/environ.py`

Key DFlash-specific additions:

- new algorithm enum choice `DFLASH`
- `speculative_dflash_block_size`
- `speculative_dflash_draft_window_size`
- `SGLANG_ENABLE_DFLASH_SPEC_V2`
- validation rules for overlap / DP attention / PP size / block size consistency

### 3. Model execution and graph layer

Must migrate:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/model_executor/memory_profiler.py`
- possibly `piecewise_cuda_graph_runner.py`

Reason:

- DFlash requires target auxiliary hidden capture
- it changes verify-mode graph handling
- it changes KV/cache sizing assumptions because the draft runner has its own behavior

### 4. Target and draft model support

Must migrate at least:

- `python/sglang/srt/models/dflash.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_moe.py`

Potentially also touched support layers under:

- `python/sglang/srt/models/deepseek_common/...`

Reason:

- target model must implement `set_dflash_layers_to_capture(...)`
- DFlash draft model has its own model class and config parsing requirements

### 5. Scheduler-side request validation

Must migrate:

- `python/sglang/srt/managers/scheduler.py`

Reason:

- DFlash rejects some request features currently allowed elsewhere
- for example logprob / hidden-state paths are not yet supported

## Recommended Migration Phases

### Phase 0: Branching and isolation

Actions:

- create dedicated DFlash branches in parent repo and SGLang submodule
- do not reuse current KT-MTP attempt branch
- keep all changes committed in small logical groups

Exit criteria:

- clean isolated DFlash working branches exist

### Phase 1: Framework skeleton port

Actions:

- port `spec_info.py`, `server_args.py`, `environ.py`
- port all `speculative/dflash_*` files
- ensure CLI accepts `DFLASH`

Validation:

- `py_compile` passes for touched files
- `python -m sglang.launch_server --help` shows DFlash args

Exit criteria:

- codebase recognizes DFlash as a valid speculative algorithm

### Phase 2: Target/draft model plumbing

Actions:

- port `models/dflash.py`
- port `set_dflash_layers_to_capture(...)` support into local target model classes
- port `model_runner.py` and `cuda_graph_runner.py` DFlash-specific capture logic

Validation:

- local load can parse draft model config
- target model runner can initialize DFlash-specific capture metadata

Exit criteria:

- target + draft model initialization reaches worker creation

### Phase 3: Minimal non-KT correctness bring-up

Actions:

- temporarily ignore KT-specific draft optimization
- focus on making raw DFlash serving run at all
- use target model route closest to upstream expected path

Validation:

- `/health = 200`
- raw `/generate` on simple deterministic prompts is correct
- `/v1/chat/completions` no longer emits broken reasoning markers

Exit criteria:

- DFlash produces semantically correct outputs on smoke tests

### Phase 4: A2 feasibility check

Actions:

- check memory with current A2-only constraint
- tune:
  - `--speculative-num-draft-tokens`
  - `--speculative-dflash-draft-window-size`
  - `--mem-fraction-static`

Validation:

- service starts without OOM
- no severe collapse in token capacity

Exit criteria:

- stable A2 launch profile exists

### Phase 5: KT compatibility evaluation

Actions:

- decide whether DFlash should remain a non-KT draft path
- only after correctness is proven, test whether any KT-specific target settings can remain unchanged

Important:

- do not assume DFlash draft can immediately reuse current KT-MTP expert routing path
- DFlash is not just "another drafter checkpoint"; it changes the speculative algorithm itself

Exit criteria:

- clear decision:
  - either "DFlash path works without KT-specific draft acceleration"
  - or "KT integration required and worth pursuing"

## High-Risk Conflict Areas

These files are especially risky because current local work already touched adjacent speculative behavior:

- `spec_info.py`
- `server_args.py`
- `model_runner.py`
- `cuda_graph_runner.py`
- `eagle_info_v2.py`
- `eagle_worker_v2.py`

Recommendation:

- do not try to merge DFlash logic into the already-modified KT-MTP attempt branch
- start from the reset copied baseline branch and port DFlash there

## What Not To Do

- Do not mix DFlash and current EAGLE KT-MTP repair work in one branch.
- Do not start by optimizing draft CPU/GPU placement before raw correctness is proven.
- Do not treat the Hugging Face model card launch command as sufficient integration proof for this repo.
- Do not assume current A2 + GPTQ_INT4 + KT stack is compatible just because upstream SGLang supports DFlash.

## Practical Expected Outcome

Most likely short-term outcomes:

1. Best case:
   - DFlash can be ported into this copied repo
   - A2 can run it with reduced draft settings
   - quality is acceptable
   - then benchmark whether it beats the current baseline

2. More likely:
   - DFlash can be integrated structurally
   - but A2 memory/performance or GPTQ/KT compatibility will still require nontrivial follow-up work

3. Failure case:
   - DFlash path depends too heavily on upstream assumptions that diverge from this KT/GPTQ deployment
   - then it should be abandoned before deep KT-specific rework

## Recommended Next Concrete Step

If work continues, the next concrete implementation step should be:

1. create `exp/dflash-integration-qw36-a2` branches
2. port only:
   - `spec_info.py`
   - `server_args.py`
   - `environ.py`
   - `speculative/dflash_*`
3. verify the local server accepts `--speculative-algorithm DFLASH`

That is the smallest meaningful checkpoint before touching deeper runner/model code.
