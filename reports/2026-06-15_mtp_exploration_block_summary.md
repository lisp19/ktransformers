# 2026-06-15 Qwen3.6 A2 MTP Exploration Summary

## Scope

Goal: enable MTP speculative decoding on the current Qwen3.6-35B-A3B GPTQ-Int4 model under the existing KT hybrid runtime on a single NVIDIA A2, without touching the baseline environment.

Result: blocked. The current SGLang + KT integration path for Qwen3.5/3.6 MTP is not fully wired through. The service can be driven much further than initial boot, but it still fails in speculative runtime paths before stable generation is available, so decode speedup and quality validation cannot be completed safely.

## Isolation And Environment Handling

- Baseline service `sglang-qw36-local.service` was stopped once to release VRAM and was not restarted.
- Baseline editable environment was not modified.
- All code and package changes were applied only to the copied experiment workspace:
  - repo copy: `/data/nvme0/ktransformers_qw36_mtp_exp`
  - conda copy: `/data/nvme0/kt_a2_qw36_mtp_exp`
- Editable installs in the copied environment were rebound to the copied repo:
  - `ktransformers -> /data/nvme0/ktransformers_qw36_mtp_exp`
  - `kt-kernel -> /data/nvme0/ktransformers_qw36_mtp_exp/kt-kernel`

## Baseline Facts Confirmed Before Experiment

- Baseline systemd unit:
  - `~/.config/systemd/user/sglang-qw36-local.service`
- Baseline runtime already used:
  - model: `/home/lsp/data/models/gptq/qwen3.6-35b-a3b`
  - env: `/data/nvme0/kt_a2_qw36_seeded_perf_base`
  - GPU UUID: `GPU-8625457e-0c7b-dd40-5d76-1f5845d811ae`
- Model package explicitly includes MTP weights:
  - `/home/lsp/data/models/gptq/qwen3.6-35b-a3b/README.md`
- Baseline decode reference from AGENTS:
  - short request decode speed about `16 tok/s`

## MTP Launch Shape Used

Experiment launch stayed aligned with baseline args and only added MTP-related deltas:

- `SGLANG_ENABLE_SPEC_V2=1`
- `--mamba-scheduler-strategy extra_buffer`
- `--speculative-algorithm EAGLE`
- `--speculative-eagle-topk 1`
- `--speculative-num-steps 3`
- `--speculative-num-draft-tokens 4`

One added env was tested and then removed:

- removed after validation: `SGLANG_MAMBA_CONV_DTYPE=float16`
  - this caused a dtype mismatch in the GDN path and is not safe for this stack

## Concrete Findings

### 1. Draft FlashInfer backend interface hole

Initial failure:

- file path: `sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- error:
  - `AttributeError: 'FlashInferMultiStepDraftBackend' object has no attribute 'on_after_cuda_graph_warmup_pass'`

Fix applied in copied `third_party/sglang`:

- file:
  - `python/sglang/srt/layers/attention/flashinfer_backend.py`
- change:
  - added `on_after_cuda_graph_warmup_pass()` passthrough on `FlashInferMultiStepDraftBackend`

Commits:

- submodule commit: `f4ab564e2`
- parent repo commit: `5c15171`

### 2. Mamba conv dtype mismatch from added env

After the first fix, the next failure was:

- file path: `sglang/srt/layers/attention/linear/gdn_backend.py`
- error:
  - `RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and BFloat16 for the source.`

Cause:

- the added optimization env `SGLANG_MAMBA_CONV_DTYPE=float16` forced conv state dtype to `Half`
- current path still produced `BFloat16` source tensors

Action:

- removed this added env from subsequent runs
- this was a configuration issue, not a baseline issue

### 3. Qwen3.5 MTP model rejected legal `input_embeds`

Next failure:

- file path: `python/sglang/srt/models/qwen3_5_mtp.py`
- error:
  - `AssertionError` on `assert input_embeds is None`

Reason:

- `model_runner.forward_extend()` can legitimately pass `input_embeds`
- current `Qwen3_5ForCausalLMMTP` implementation rejected that path

Fix applied:

- changed the model to accept passed `input_embeds`
- fallback remains `forward_batch.mm_input_embeds`
- final fallback remains token embedding lookup

Commits:

- submodule commit: `b2caa0418`
- parent repo commit: `2edb535`

### 4. Qwen3.5 MTP multimodal guard was too strict

Next failure on later draft extend:

- file path: `python/sglang/srt/models/qwen3_5_mtp.py`
- error:
  - `AssertionError` on `assert input_embeds is not None`

Reason:

- `contains_mm_inputs()` could still be true in this path while no actual multimodal embeds were provided
- the branch should only activate when `input_embeds` is present

Fix applied:

- narrowed the condition so the multimodal splice branch runs only when `input_embeds is not None`

Commits:

- submodule commit: `b60ba8fe0`
- parent repo commit: `5efc5d5`

### 5. Current hard block: `mrope_positions` type mismatch in speculative decode path

After the above fixes, the runtime progressed further, including:

- target model weight load
- target CUDA graph capture
- MTP draft model weight load
- draft CUDA graph capture
- uvicorn startup
- `GET /model_info -> 200`

But the scheduler still failed before healthy stable generation:

- file path:
  - `python/sglang/srt/model_executor/forward_batch_info.py`
- call chain:
  - `eagle_worker_v2.py`
  - `eagle_info_v2.py`
  - `ForwardBatch.init_new()`
  - `_compute_mrope_positions()`
- error:
  - `TypeError: expected Tensor as element 0 in argument 0, but got list`

This is the current block condition.

## Why This Is Considered A Hard Block

At this point the experiment is no longer failing on a single obvious integration seam. It has already exposed multiple independent missing/incorrect code paths specific to:

- FlashInfer draft warmup callbacks
- Mamba/GDN dtype coordination
- Qwen3.5 MTP extend handling of `input_embeds`
- Qwen3.5 MTP multimodal guard behavior
- speculative decode `mrope_positions` construction

This pattern indicates the present SGLang branch does not yet have a complete, stable Qwen3.5/3.6 MTP speculative path for this KT deployment. Continuing would shift from "enable MTP and validate speedup" into upstream-style feature repair across several subsystems.

That exceeds the "simple adaptation and debug" bar for this phase.

## What Was Not Achieved

- No stable `/health = 200` steady state under real speculative generation
- No OpenAI-compatible generation validation with preserved output quality
- No decode throughput measurement with MTP enabled
- No proof of improvement over the `~16 tok/s` baseline

## Current Local Git State Of Experiment Repo

Parent repo recent commits:

- `5efc5d5 [chore]: update sglang submodule for qwen mtp mm guard`
- `2edb535 [chore]: update sglang submodule for qwen mtp input embeds`
- `5c15171 [chore]: update sglang submodule for mtp draft warmup`

Copied `third_party/sglang` recent commits:

- `b60ba8fe0 fix: relax qwen3.5 mtp multimodal embed guard`
- `b2caa0418 fix: accept input embeds in qwen3.5 mtp`
- `f4ab564e2 fix: forward flashinfer draft warmup hook`

## Recommendation For Next Stage

If a next goal is created, it should be framed explicitly as one of these:

1. Repair upstream Qwen3.5/3.6 MTP speculative runtime compatibility in copied SGLang until stable generation works.
2. Abandon current SGLang MTP path on this branch and evaluate an alternate engine/path for the same model package.
3. Keep KT baseline path unchanged and pursue non-MTP decode optimization instead.

For option 1, the next debugging entry point should be:

- `python/sglang/srt/model_executor/forward_batch_info.py`
- investigate `_compute_mrope_positions()` inputs in the speculative draft/decode path, especially where list-valued entries are introduced before `torch.cat`.
