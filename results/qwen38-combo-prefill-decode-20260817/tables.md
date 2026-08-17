## Phase-separated vLLM benchmark — Qwen3.8-27B COMBO (Sergio gptqmodel quant on our 5-patch stack)

Tested stack: vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, `vllm-xpu-kernels 0.1.12`, C1, `n=5`, scheduler budget 4096, fp8 KV, context 131328, prefix cache enabled with zero hit delta, configured 230 W cap, MTP4 + S+M1 draft-INT4 patches (PX2: ptr_wrap + gdn_split_mixed). Status: E2 self-reported. Snapshot 15/17 cells.

### Cold input rate (actual input tokens / TTFT, tok/s)

| Mode | p512 | p2048 | p4096 | p6144 | p8192 | Full p131071 |
|---|---:|---:|---:|---:|---:|---:|
| MTP4 COMBO | 1,806 | 1,850 | 1,782 | 1,752 | 1,690 | 711 |

### Decode at p512 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 92.9 | 113.0 | 114.0 | 74.3 |

### Decode at p8192 (client post-first tok/s)

| Mode | g32 | g128 | g256 | g512 |
|---|---:|---:|---:|---:|
| MTP4 COMBO | 98.6 | 105.6 | 79.4 | 65.6 |

### Matched historical control (p9445/g128)

| Mode | Client post-first median (tok/s) |
|---|---:|
| MTP4 COMBO | 104.9 |

### Full-context decode

| Mode | p130944/g128 (tok/s) | MTP accept | p130560/g512 (tok/s) | MTP accept |
|---|---:|---:|---:|---:|
| MTP4 COMBO | in progress | n/a | pending | n/a |


### Detailed per-cell (medians, n=5)

| Cell | TTFT (s) | E2E (s) | input/TTFT (tok/s) | post-first (tok/s) | MTP accept % |
---| | |---| | |---| | |---| | |---| | |---|
| prefill-p512 | *pending* | | | | |
| prefill-p2048 | *pending* | | | | |
| prefill-p4096 | *pending* | | | | |
| prefill-p6144 | *pending* | | | | |
| prefill-p8192 | *pending* | | | | |
| decode-p512-g32 | 0.3 | 0.6 | 1,648 | 92.9 | 80.8% |
| decode-p512-g128 | 0.3 | 1.4 | 1,664 | 113.0 | 94.1% |
| decode-p512-g256 | 0.3 | 2.5 | 1,663 | 114.0 | 80.6% |
| decode-p512-g512 | 0.3 | 7.2 | 1,661 | 74.3 | 61.5% |
| decode-p8192-g32 | 4.9 | 5.2 | 1,668 | 98.6 | 83.8% |
| decode-p8192-g128 | 4.9 | 6.1 | 1,669 | 105.6 | 95.1% |
| decode-control-p9445-g128 | 5.7 | 6.9 | 1,655 | 104.9 | 91.5% |
| decode-p8192-g256 | 4.9 | 8.1 | 1,671 | 79.4 | 65.9% |
| decode-p8192-g512 | 4.9 | 12.7 | 1,669 | 65.6 | 49.5% |
| prefill-full-p131071 | *pending* | | | | |
| decode-full-p130944-g128 | *pending* | | | | |
| decode-full-p130560-g512 | *pending* | | | | |

Input rate includes request scheduling + first-token work; decode is client-observed, not engine-native.
Comparing stacks see COMPARATIVA-STACKS-20260817.md in this directory.
