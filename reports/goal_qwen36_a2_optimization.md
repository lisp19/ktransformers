# Qwen3.6 35B A3B on A2 Optimization Record

Date: 2026-06-06

## Final Runtime Path

- Service file: `/home/lsp/.config/systemd/user/sglang-qw36-local.service`
- Working directory: `/home/lsp/a2_kt/ktransformers`
- Python environment: `/data/nvme0/kt_a2_qw36_seeded_base`
- Repo Python override: `PYTHONPATH=/home/lsp/a2_kt/ktransformers/third_party/sglang/python`
- CUDA allocator tweak: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- GPU binding: `CUDA_VISIBLE_DEVICES=GPU-8625457e-0c7b-dd40-5d76-1f5845d811ae`

## New Baseline Environment

- Source clone prefix: `/home/lsp/a2_kt/ktransformers/kt_a2_env`
- New baseline prefix: `/data/nvme0/kt_a2_qw36_seeded_base`
- Creation method: `conda create --prefix /data/nvme0/kt_a2_qw36_seeded_base --clone /home/lsp/a2_kt/ktransformers/kt_a2_env -y`

## Effective Optimization Method

1. Keep `ktransformers` hybrid inference path on A2 with `kt_num_gpu_experts=48`.
2. Use `frequency` placement with a recorded expert-distribution file instead of the previous uniform fallback.
3. Use `expandable_segments:True` to keep the seeded-frequency layout stable at long context.
4. Keep the KT hybrid remap fast path in `third_party/sglang`.
5. Fix expert distribution recorder dump on single-rank runs by skipping unnecessary `all_reduce`.

## Activation Statistics File

- `reports/expert_distribution/expert_distribution_recorder_1780728976.5500257.pt`

This file contains:

- `logical_count`: `(64, 40, 256)` `torch.int32`

It is used as:

- `--init-expert-location /home/lsp/a2_kt/ktransformers/reports/expert_distribution/expert_distribution_recorder_1780728976.5500257.pt`

## Effective Code Commits

### Superproject

- `4211223` `[perf]: wire seeded frequency placement into copy service`
- `44641b3` `[perf]: update sglang submodule for kt hybrid remap fast path`
- `97d2ee2` `[chore]: use repo sglang python in copy service template`
- `7b9d9cb` `[chore]: add workspace copy service template for qwen3.6 a2`

### `third_party/sglang`

- `5cc8b3b1f` `[fix]: skip recorder all_reduce on single-rank runs`
- `4b16c98dc` `[perf]: skip redundant expert remap in kt hybrid moe`

## Acceptance Benchmark Result

Baseline file:

- `reports/current_service_baseline_3r.json`

Optimized file:

- `reports/copy_lane_perf_frequency_seeded_allocfix_3r.json`

### Delta vs Baseline

- `short_lt50`
  - prefill: `24.82 -> 36.52` (`+47.14%`)
  - decode: `14.78 -> 17.63` (`+19.34%`)
- `prompt_500`
  - prefill: `38.55 -> 58.10` (`+50.70%`)
  - decode: `14.99 -> 18.81` (`+25.49%`)
- `prompt_1000`
  - prefill: `37.60 -> 54.42` (`+44.75%`)
  - decode: `14.59 -> 18.57` (`+27.29%`)
- `prompt_3000`
  - prefill: `37.84 -> 53.13` (`+40.38%`)
  - decode: `14.65 -> 19.84` (`+35.46%`)

## Runtime Validation Notes

- Endpoint successfully starts on A2.
- Long context target remains configured at `262144`.
- A2 memory usage reaches about `14906 / 15356 MiB` during service runtime, about `97%`.
- Output generation returns meaningful Chinese text.

## Rejected Intermediate Paths

- `kt_max_deferred_experts_per_token=1`
  - Improved decode somewhat, but not enough to beat the final seeded-frequency solution.
- `kt_max_deferred_experts_per_token=2`
  - Strong decode gain, but long-context prefill regressed relative to the best code-only path.
- Additional CPU submit fast paths tested in KT wrapper / `experts_base.py`
  - Did not beat the final seeded-frequency configuration and were reverted from the selected runtime path.
