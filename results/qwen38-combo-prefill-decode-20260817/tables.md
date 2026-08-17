## Phase-separated vLLM benchmark — Qwen3.8-27B COMBO (gptqmodel quant on the 5-patch stack)

Tested stack: vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`, C1, `n=5`, scheduler budget 4096, context 131328, fp8 KV, prefix cache enabled with zero hit delta, configured 230 W cap, MTP4 with S+M1 draft-INT4 patches. Status: E2 self-reported.

### Cold input rate (actual input tokens / TTFT, tok/s)

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| MTP4 COMBO | 1,806 | 1,850 | 1,782 | 1,752 | 1,690 | 711 |

### Decode at p512 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 92.88 | 112.98 | 114.00 | 74.32 |

### Decode at p8192 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 98.65 | 105.59 | 79.39 | 65.59 |

### Matched historical control (p9445/g128)

| Mode | Client post-first median (tok/s) |
|---|---:|
| MTP4 COMBO | 104.86 |

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 60.35 | 79.2% | 47.26 | 59.7% |

Input rate includes request scheduling and first-token work; decode is client-observed, not engine-native. Exact output rows use the requested completion length.

### Detailed per-cell (medians, n=5)

| Cell | TTFT (s) | E2E (s) | input/TTFT (tok/s) | post-first (tok/s) | TPOT (ms) | MTP accept % |
|---|---:|---:|---:|---:|---:|---:|
| prefill-p512 | 0.28 | 0.28 | 1,806 | n/a | n/a | n/a% |
| prefill-p2048 | 1.11 | 1.11 | 1,850 | n/a | n/a | n/a% |
| prefill-p4096 | 2.30 | 2.30 | 1,782 | n/a | n/a | n/a% |
| prefill-p6144 | 3.51 | 3.51 | 1,752 | n/a | n/a | n/a% |
| prefill-p8192 | 4.85 | 4.85 | 1,690 | n/a | n/a | n/a% |
| decode-p512-g32 | 0.31 | 0.64 | 1,648 | 92.88 | 10.8 | 80.77% |
| decode-p512-g128 | 0.31 | 1.43 | 1,664 | 112.98 | 8.9 | 94.07% |
| decode-p512-g256 | 0.31 | 2.54 | 1,663 | 114.00 | 8.8 | 80.59% |
| decode-p512-g512 | 0.31 | 7.18 | 1,661 | 74.32 | 13.5 | 61.51% |
| decode-p8192-g32 | 4.91 | 5.23 | 1,668 | 98.65 | 10.1 | 83.78% |
| decode-p8192-g128 | 4.91 | 6.11 | 1,669 | 105.59 | 9.5 | 95.15% |
| decode-control-p9445-g128 | 5.71 | 6.92 | 1,655 | 104.86 | 9.5 | 91.49% |
| decode-p8192-g256 | 4.90 | 8.12 | 1,671 | 79.39 | 12.6 | 65.91% |
| decode-p8192-g512 | 4.91 | 12.70 | 1,669 | 65.59 | 15.2 | 49.53% |
| prefill-full-p131071 | 184.39 | 184.39 | 711 | n/a | n/a | n/a% |
| decode-full-p130944-g128 | 184.36 | 186.51 | 710 | 60.35 | 16.6 | 79.17% |
| decode-full-p130560-g512 | 183.75 | 194.59 | 711 | 47.26 | 21.2 | 59.69% |
