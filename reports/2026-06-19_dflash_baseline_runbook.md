# DFlash Baseline Runbook

Date: 2026-06-19

Scope:

- copied experiment repo only
- copied conda env only
- A2 only
- does not replace or modify the original baseline environment

## Code baseline

- parent repo branch: `exp/dflash-integration-qw36-a2`
- `sglang` submodule branch: `exp/dflash-integration-qw36-a2`
- parent repo commit: see branch tip when this document is committed
- `sglang` submodule commit: see branch tip when this document is committed

Key effective fixes included in this baseline:

- DFlash stack imported from upstream PR 20547
- local KT compatibility restoration
- `block_size=1` fallback path for DFlash debug correctness
- shared `qwen3-thinking` chat-path fix allowing `enable_thinking=false`

## Runtime baseline

This is the currently verified DFlash serving shape that should be treated as the
`DFlash baseline` for this experiment lane:

- `DFLASH`
- `speculative_num_draft_tokens=16`
- `speculative_dflash_block_size=16`
- `speculative_dflash_draft_window_size=16`
- `--disable-cuda-graph`
- `--skip-server-warmup`
- A2 only

Notes:

- `--disable-cuda-graph` is part of the currently working DFlash baseline.
- `chat/completions` should pass `chat_template_kwargs.enable_thinking=false` when the
  client wants normal assistant `content` instead of separated reasoning output.

## Launch command

```bash
env \
  CUDA_VISIBLE_DEVICES=GPU-8625457e-0c7b-dd40-5d76-1f5845d811ae \
  PYTHONPATH=/data/nvme0/ktransformers_qw36_mtp_exp/third_party/sglang/python \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TVM_FFI_CACHE_DIR=/data/nvme0/tvm-ffi-perf-base \
  /data/nvme0/kt_a2_qw36_mtp_exp/bin/python -u -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 30001 \
    --skip-server-warmup \
    --model /home/lsp/data/models/gptq/qwen3.6-35b-a3b \
    --kt-weight-path /home/lsp/data/models/gptq/qwen3.6-35b-a3b \
    --kt-cpuinfer 16 \
    --kt-threadpool-count 1 \
    --kt-num-gpu-experts 0 \
    --kt-method GPTQ_INT4 \
    --kt-gpu-prefill-token-threshold 16384 \
    --kt-enable-dynamic-expert-update \
    --attention-backend flashinfer \
    --trust-remote-code \
    --mem-fraction-static 0.92 \
    --chunked-prefill-size 2048 \
    --max-total-tokens 32768 \
    --reasoning-parser qwen3-thinking \
    --tool-call-parser qwen \
    --max-running-requests 1 \
    --watchdog-timeout 600 \
    --enable-mixed-chunk \
    --tensor-parallel-size 1 \
    --enable-p2p-check \
    --weight-loader-disable-mmap \
    --kv-cache-dtype fp8_e4m3 \
    --kt-expert-placement-strategy frequency \
    --init-expert-location /data/nvme0/ktransformers_qw36_mtp_exp/reports/expert_distribution/expert_distribution_recorder_1780728976.5500257.pt \
    --disable-shared-experts-fusion \
    --disable-cuda-graph \
    --served-model-name qwen3.6-35b-a3b \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /home/lsp/data/models/gptq/Qwen3.6-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --speculative-dflash-block-size 16 \
    --speculative-dflash-draft-window-size 16 \
    --mamba-scheduler-strategy extra_buffer
```

## Verified checks

### Health

```bash
curl -sS http://127.0.0.1:30001/health
```

Expected:

- HTTP `200`

### Raw generation

```bash
curl -sS -X POST http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"只回答最终结果：13乘以17等于多少？",
    "sampling_params":{"max_new_tokens":16,"temperature":0}
  }'
```

Expected:

- final visible answer `221`

### OpenAI chat path

```bash
curl -sS -X POST http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "messages":[{"role":"user","content":"只回答最终结果：13乘以17等于多少？"}],
    "temperature":0,
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Expected:

- `choices[0].message.content == "221"`
- `choices[0].message.reasoning_content == null`

### OpenAI JSON probe

```bash
curl -sS -X POST http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "messages":[{"role":"user","content":"请直接输出一个合法JSON：{\"result\":221}"}],
    "temperature":0,
    "max_tokens":128,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Expected:

- `choices[0].message.content == "{\"result\":221}"`

## Artifact paths

- DFlash model directory:
  `/home/lsp/data/models/gptq/Qwen3.6-35B-A3B-DFlash`
- Main experiment repo:
  `/data/nvme0/ktransformers_qw36_mtp_exp`
- Main experiment env:
  `/data/nvme0/kt_a2_qw36_mtp_exp`

## Important limitations

- This baseline is the current `DFlash` experiment baseline, not the original
  production baseline.
- It intentionally does not replace the original baseline environment.
- It currently depends on `--disable-cuda-graph`.
