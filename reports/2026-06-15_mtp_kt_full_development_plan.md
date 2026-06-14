# 2026-06-15 KT MTP Full Development Plan

## Goal

On top of the copied experiment repo, implement a working KT-kernel-based MTP path for Qwen3.6 on a single A2 with these properties:

1. MTP-related draft components must not be forced to remain fully resident on GPU.
2. The MTP MoE path must allow part of the draft experts to stay on CPU.
3. OpenAI-compatible output quality must be validated before claiming success.
4. Baseline environment and baseline service remain untouched.

This document is the implementation plan for the next phase. It is not a claim that the feature already works.

## Current Facts From The Repo

### 1. Draft worker currently forbids KT on purpose

Current speculative worker construction wraps draft worker init under:

- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/eagle_worker.py`
- `python/sglang/srt/speculative/standalone_worker_v2.py`

The common mechanism is:

- `python/sglang/srt/layers/moe/utils.py`
- `speculative_kt_ep_disabled_context()`

And `create_kt_config_from_server_args()` returns `None` when that flag is set:

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`

So upstream behavior is explicitly:

- target model can use KT
- speculative draft model must use pure GPU MoE

This is incompatible with the current goal.

### 2. KT wrapper already supports BF16 MoE experts

The copied code already contains a BF16/unquantized expert path:

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
  - `WEIGHT_NAMES_BF16`
  - `_prepare_weight_bf16()`
  - BF16 expert copy/update helpers
- `kt-kernel/ext_bindings.cpp`
  - `AMXBF16_MOE`
  - `AVX2BF16_MOE`
- `kt-kernel/README.md`
  - `--kt-method BF16`

This matters because the Qwen3.6 MTP experts are BF16, not GPTQ INT4.

### 3. Qwen3.6 MTP weights are a separate one-layer branch

From the model package:

- `/home/lsp/data/models/gptq/qwen3.6-35b-a3b/config.json`
  - `text_config.mtp_num_hidden_layers = 1`
  - `text_config.mtp_use_dedicated_embeddings = false`
  - `tie_word_embeddings = false`
- `/home/lsp/data/models/gptq/qwen3.6-35b-a3b/model.safetensors.index.json`
  - MTP weights live in `mtp.safetensors`
  - experts are under `mtp.layers.0.mlp.experts.*`
  - extra MTP tensors include `mtp.fc.weight`, `mtp.pre_fc_norm_*`, attention weights, and `mtp.norm.weight`

So the draft branch is not a generic external model path. It is a one-layer MoE branch embedded in the same checkpoint family.

### 4. Draft worker currently shares target embed/lm_head

Current EAGLE path does:

- `python/sglang/srt/speculative/eagle_worker_v2.py`
  - `init_lm_head()`
  - `self.draft_runner.model.set_embed_and_head(embed, head)`

This means the draft path inherits GPU-resident target embedding/lm_head by default. That directly conflicts with the new requirement that MTP-related lm head / layer / draft model should not be pinned to GPU by construction.

### 5. Prior debugging already exposed multiple missing runtime paths

The previous exploration hit and partially fixed:

- FlashInfer draft warmup callback missing
- Mamba conv dtype mismatch
- Qwen3.5 MTP `input_embeds` handling bugs
- Qwen3.5 MTP multimodal guard bug

The current blocking runtime issue is:

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `_compute_mrope_positions()`
- draft decode path eventually feeds a list into `torch.cat`

This means the next phase must treat the work as a real implementation project, not a parameter-tuning task.

## Required Design Direction

### Design Principle A: Draft KT must be independently configurable

The draft branch cannot safely reuse the target KT configuration 1:1.

Reasons:

- target experts are GPTQ INT4
- MTP experts are BF16
- target has 40 MoE layers
- MTP branch has 1 MoE layer
- draft memory budget is tighter than target budget

Therefore the draft path needs its own KT config surface, even if defaults inherit from the target config.

### Design Principle B: GPU residency must be explicit and revocable

For the draft branch, “not fully resident on GPU” should mean:

- no unconditional share of target lm_head / embeddings onto the draft model
- no unconditional all-expert GPU materialization for the draft MoE layer
- non-expert draft weights may be staged to GPU for execution, but must have a CPU-master / non-resident mode

This is different from “never touched by GPU”. The draft branch still has to execute on GPU to keep speculative decoding useful.

### Design Principle C: Quality validation must be first-class

The implementation is not complete if it only boots and produces tokens.

Minimum acceptance for final completion:

- `/health` steady 200
- OpenAI `/v1/chat/completions` returns meaningful content
- deterministic smoke prompts pass rule-based checks
- no obvious reasoning collapse / repetition / malformed JSON on structured prompts
- decode speed improves over the non-MTP KT baseline on short requests

## Proposed Implementation Workstreams

## Workstream 1: Draft-Specific KT Control Plane

### Objective

Allow speculative draft workers to opt into KT instead of being hard-disabled.

### Files To Change

- `python/sglang/srt/layers/moe/utils.py`
- `python/sglang/srt/speculative/eagle_worker_v2.py`
- `python/sglang/srt/speculative/eagle_worker.py`
- `python/sglang/srt/speculative/standalone_worker_v2.py`
- `python/sglang/srt/server_args.py`

### Concrete Changes

1. Replace unconditional `speculative_kt_ep_disabled_context()` usage with a conditional path.
2. Add draft-specific additive knobs, for example:
   - `--kt-mtp-enable`
   - `--kt-mtp-method`
   - `--kt-mtp-num-gpu-experts`
   - `--kt-mtp-gpu-experts-ratio`
   - `--kt-mtp-share-embed-head`
3. Build a draft-local `ServerArgs` clone instead of mutating the same object used by the target worker.
4. Default behavior must stay backward compatible:
   - if `--kt-mtp-enable` is not set, draft still uses pure GPU MoE as today

### Why This Is First

Without this workstream, every later KT-specific draft change is dead code because draft worker creation still suppresses KT globally.

## Workstream 2: KTConfig Support For MTP Layer Prefixes

### Objective

Make KT able to load MTP expert weights from `mtp.layers.0.*` instead of only base-model `model.language_model.layers.*`.

### Files To Change

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- `kt-kernel/python/experts_base.py`
- if needed: KT loader helpers under `kt-kernel/python/`

### Concrete Changes

1. Extend `KTConfig` with prefix metadata, e.g.:
   - `weight_prefix`
   - `is_mtp_layer`
   - `cpu_weight_dtype`
2. Update `create_kt_config_from_server_args()` so draft workers derive masks for:
   - `num_moe_layers = 1`
   - weight prefix `mtp.layers.0`
3. Keep current target behavior unchanged.
4. Reuse existing BF16 expert loading path in `kt_ep_wrapper.py` for MTP experts.
5. Add CPU/GPU placement logging for the MTP layer separately from target MoE layers.

### Expected Outcome

Draft MTP experts can be split:

- hot experts on GPU
- cold experts on CPU via KT BF16 backend

## Workstream 3: Draft Non-Expert Residency Strategy

### Objective

Prevent the draft branch from forcing all of its non-expert weights to live on GPU permanently, especially embedding and lm_head.

### Current Conflict

Today `eagle_worker_v2.init_lm_head()` shares target embed/lm_head into the draft model. That is simple, but it permanently ties draft residency to target GPU weights.

### Proposed Solution

Adopt a standalone-draft-style path for KT MTP:

1. Do not share target embed/lm_head by default when `--kt-mtp-enable` is active.
2. Use a draft-specific residency strategy:
   - CPU-master copy for draft embed/lm_head and non-MoE block weights
   - staged GPU resume before draft forward
   - release/pause after draft use
3. Prefer existing memory-saver infrastructure where possible:
   - `python/sglang/srt/utils/torch_memory_saver_adapter.py`
   - `model_runner.load_model()` already supports CPU backup flags
4. If memory-saver is insufficient, introduce an explicit draft offload manager for:
   - embed_tokens
   - lm_head
   - draft linear-attn/full-attn layer params
   - `fc`, `pre_fc_norm_*`, `norm`

### Important Constraint

This workstream must not silently degrade quality by replacing full-vocab logits with an approximate shortlist. Any CPU/offloaded `lm_head` design still has to preserve the exact next-token distribution used by the draft path.

### Preferred Runtime Behavior

- resident on GPU:
  - current-step activations
  - selected draft GPU experts
  - temporary staged non-expert tensors only while executing draft forward
- resident on CPU:
  - draft master weights
  - all CPU experts

## Workstream 4: Complete Qwen3.5/3.6 MTP Runtime Compatibility

### Objective

Repair the remaining speculative runtime gaps until stable generation works end-to-end.

### Known Items Already Touched

- draft FlashInfer warmup callback
- `input_embeds` acceptance in `qwen3_5_mtp.py`
- multimodal embed guard in `qwen3_5_mtp.py`

### Next Known Blocker

- `python/sglang/srt/model_executor/forward_batch_info.py`
- `_compute_mrope_positions()`
- draft decode path currently feeds a list into `torch.cat`

### Concrete Deliverables

1. Fix `mrope_positions` construction for draft extend / draft decode.
2. Add regression coverage for:
   - prefill -> draft extend
   - decode -> draft extend
   - multimodal flag present but no draft mm embeds
3. Ensure the draft path still works after KT is re-enabled for the MTP layer.

## Workstream 5: Quality Validation Pipeline

### Objective

Make output-quality validation reproducible and local, instead of ad-hoc terminal checks.

### Artifacts To Add

- `tools/mtp_quality_cases.json`
- `tools/validate_serving_quality.py`
- a saved non-MTP reference output file under `reports/`

### Validation Method

1. Bring up the copied non-MTP KT service and record reference outputs at `temperature=0`.
2. Use the same prompt set against the KT+MTP service.
3. Enforce rule-based checks:
   - non-empty output
   - must-contain facts for deterministic tasks
   - must-not-contain repetition markers / refusal noise
   - structured JSON prompts parse successfully
4. Store all outputs and metrics to JSON under `reports/`.

### Prompt Categories

- Chinese short response
- factual extraction from a provided paragraph
- arithmetic with exact answer
- short code generation
- structured JSON response

### Completion Gate

MTP is not accepted unless the quality script passes on all required cases.

## Workstream 6: Performance Validation

### Objective

Prove that MTP improves decode throughput on this A2 setup.

### Existing Reusable Tool

- `tools/compare_serving_perf.py`

### Required Comparison

1. Copied non-MTP KT service
2. KT+MTP draft-hybrid service

### Minimum Metrics

- short-request decode tps
- prompt_500 decode tps
- TTFT
- whether output remains meaningful

### Acceptance

For the final goal, short-request decode must beat the current baseline reference of about `16 tok/s` without obvious quality degradation.

## Recommended Execution Order

1. Workstream 1
   Enable draft KT control plane.
2. Workstream 2
   Make KT load `mtp.layers.0.*` BF16 experts with per-expert CPU/GPU placement.
3. Workstream 4
   Finish the speculative runtime compatibility fixes until real generation is stable.
4. Workstream 3
   Remove unconditional GPU residency for draft embed/lm_head/non-expert weights.
5. Workstream 5
   Freeze quality validation.
6. Workstream 6
   Measure speedup and decide final tuning.

## Suggested File-Level Task Breakdown

### Phase A: Enable draft KT

- `python/sglang/srt/server_args.py`
  add draft-specific KT knobs
- `python/sglang/srt/layers/moe/utils.py`
  conditional KT disable policy
- `python/sglang/srt/speculative/eagle_worker_v2.py`
  build draft-local args and stop unconditional GPU-only KT bypass

### Phase B: MTP BF16 expert loading

- `python/sglang/srt/layers/moe/kt_ep_wrapper.py`
  add MTP prefix + one-layer placement support
- `kt-kernel/python/experts_base.py`
  expose / document draft-specific weight path semantics if needed

### Phase C: Runtime correctness

- `python/sglang/srt/model_executor/forward_batch_info.py`
  fix `mrope_positions`
- `python/sglang/srt/speculative/eagle_info_v2.py`
  inspect inputs feeding `ForwardBatch.init_new()`
- `python/sglang/srt/models/qwen3_5_mtp.py`
  keep draft extend semantics correct

### Phase D: Draft non-resident lm_head / layer

- `python/sglang/srt/speculative/eagle_worker_v2.py`
  stop unconditional share of target embed/head
- `python/sglang/srt/models/qwen3_5_mtp.py`
  support staged embed/head injection cleanly
- if needed:
  - `python/sglang/srt/layers/logits_processor.py`
  - draft-specific exact full-vocab logits path with staged weights

## Risks

### High Risk

- exact full-vocab draft `lm_head` without permanent GPU residency may still be too slow
- draft KT re-enable may expose new unsupported paths in MoE or attention code

### Medium Risk

- MTP BF16 expert path may require extra loader glue for `mtp.safetensors`
- `mrope_positions` fix may uncover additional multimodal or draft-shape assumptions

### Low Risk

- quality validation tooling
- performance comparison tooling reuse

## Definition Of Done

The feature is complete only if all of the following are true:

1. Draft worker can run with KT enabled.
2. Draft MTP MoE layer supports mixed CPU/GPU expert placement.
3. Draft embed/lm_head/non-expert layer are not forced to remain permanently GPU-resident.
4. Service reaches stable `/health = 200`.
5. OpenAI `/v1/chat/completions` returns meaningful outputs.
6. Local quality validation passes.
7. Decode throughput is improved over the current non-MTP KT baseline.

## Immediate Next Concrete Task

Start with Workstream 1 + Workstream 2 together:

- add draft-specific KT knobs
- allow draft worker to bypass `speculative_kt_ep_disabled_context()` conditionally
- teach `kt_ep_wrapper` to treat the MTP layer as a BF16 one-layer KT MoE with prefix `mtp.layers.0`

That is the smallest change set that creates a real KT-backed MTP draft path instead of only fixing runtime glue around a pure-GPU draft worker.
