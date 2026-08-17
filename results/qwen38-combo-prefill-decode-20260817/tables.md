# Qwen3.8-27B COMBO — phase-separated benchmark (Sergio gptqmodel quant on our 5-patch stack)

**Status: 14/17 cells complete (partial).** Full-context cells (p131071 prefill, p130944/g128 and p130560/g512 decode) pending — being filled live.

Tested stack: vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`, C1, `n=5`, scheduler budget 4096, context 131328, fp8 KV, prefix cache on with zero hit delta, 230 W cap, MTP4 with S+M1 draft-INT4 patches. Self-reported E2.

## Cold input rate (actual input tokens / TTFT, tok/s)

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| MTP4 COMBO | 1,806 | 1,850 | 1,782 | 1,752 | 1,690 | p131071 (in progress) |

## Decode at p512 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 92.9 | 113.0 | 114.0 | 74.3 |

## Decode at p8192 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 98.6 | 105.6 | 79.4 | 65.6 |

## Matched historical control (p9445/g128)

| Mode | Client post-first median (tok/s) |
|---|---:|
| MTP4 COMBO | 104.9 |

## Full-context decode

| Mode | p130944/g128 | MTP accept | p130560/g512 | MTP accept |
|---|---:|---:|---:|---:|
| MTP4 COMBO | pending | — | pending | — |

