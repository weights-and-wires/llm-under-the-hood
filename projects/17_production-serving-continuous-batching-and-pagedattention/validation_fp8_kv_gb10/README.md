# Hardware validation: fp8 KV cache + speculative decoding (NVIDIA GB10)

The 2026 update to Projects 14 and 17 discusses two serving levers — **fp8 KV-cache
quantization** and **speculative decoding** — and the trade-offs of combining them.
This folder is the receipt: the labs run end-to-end on real hardware.

- **Hardware:** NVIDIA GB10 (DGX Spark), 128 GB unified memory
- **Engine:** vLLM v0.19.2rc1 (cu130 nightly), `vllm/vllm-openai` container
- **Model:** `Qwen/Qwen3-8B`
- **Load:** 200 prompts, random 1024 in / 256 out, via `vllm bench serve`

Open the PNGs in [`graphs/`](graphs) for the charts; raw per-config results are in
[`data/`](data) (`*.json` = `vllm bench serve` output, `*.kv` = the KV-budget line vLLM
prints at startup, `*.metrics` = the Prometheus spec-decode counters).

## What reproduced

| Claim (Projects 14 & 17) | Measured |
|---|---|
| fp8 KV cache ≈ 2× the token budget of bf16 | **2.07×** KV tokens (619,504 vs 299,728) and 2.07× max concurrency; sweep holds 2.0–2.06× across memory fractions |
| A causal (n-gram) drafter composes with fp8 KV | fp8 KV + n-gram serves cleanly and is the best config — **1.9×** throughput, **3.8×** faster per-token latency, **71%** draft acceptance |
| Speculation is the dominant decode-latency win | median TPOT 84.5 → 22.5 ms |
| Non-causal (DFlash) drafter + fp8 = the coupons that won't stack | **Not executable** — DFlash is not in stock vLLM (issue #41559), so the incompatibility itself cannot be driven; the composing contrast it is paired against is confirmed instead |

## How to read the two lines vLLM prints

You don't have to hunt for an out-of-memory error to see the fp8 budget win. vLLM reports
it at startup:

```
GPU KV cache size: 619,504 tokens
Maximum concurrency for 32,768 tokens per request: 18.91x
```

Serve once with `--kv-cache-dtype fp8` and once with `--kv-cache-dtype auto` (bf16) and
compare those two lines — the fp8 number is about twice the bf16 one.

## Honest caveats

- **Absolute tokens/sec are GB10-specific.** The GB10's unified LPDDR5X is bandwidth-limited,
  so decode throughput is well below an H100's. The **ratios** are the finding, not the raw numbers.
- **Benchmark TTFT is queue-saturated by design** (all 200 prompts fired at once), so read it as
  an effective-capacity signal, not single-request latency.

## Reproduce

```bash
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8  --max-model-len 32768 --gpu-memory-utilization 0.5
vllm serve Qwen/Qwen3-8B --kv-cache-dtype auto --max-model-len 32768 --gpu-memory-utilization 0.5
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
# then, against each: vllm bench serve --model Qwen/Qwen3-8B --dataset-name random \
#   --num-prompts 200 --random-input-len 1024 --random-output-len 256 --save-result
```
