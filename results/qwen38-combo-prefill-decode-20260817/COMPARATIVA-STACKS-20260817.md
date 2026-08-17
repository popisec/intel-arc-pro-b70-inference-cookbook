# Comparativa de stacks — 17-ago-2026 · RESULTADO ADOPTADO: COMBO

> **Decisión del proyecto (17-ago-2026):** el **COMBO** — modelo de Sergio
> (`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, quant **gptqmodel 7.3.2**,
> rev `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`) servido en **NUESTRO stack**
> (vLLM `0.26.1rc1.dev457+gc810e5ee9.xpu`, S+M1 + gdn_split + ptr_wrap + MBT 4096) —
> es la config oficial del servidor de referencia (contenedor dedicado, puerto 8010).
> Mediciones todas del 16-17-ago-2026 en la una sola Intel Arc Pro B70, 230 W,
> harness propio, C1, n=5, A/B térmico.

## 1. Los tres stacks comparados

| | **COMBO** (adoptado) | Sergio (todo suyo) | Nuestro (todo nuestro) |
|---|---|---|---|
| Imagen | `2c427ef477da` (0.26.1) | `f01e24f6c7ff` (0.27.2) | `2c427ef477da` (0.26.1) |
| kernels | 0.1.12 | 0.1.12.3 | 0.1.12 |
| Modelo | **gptqmodel 7.3.2** (Sergio) | gptqmodel 7.3.2 | AutoRound 512/4 (propia) |
| Patches | 5 (S+M1+gdn+ptr_wrap+boundary+nightly) | 2 (nightly+boundary) | 5 igual | 
| MBT | 4096 | 8192 | 4096 |
| Prefix-cache | ON | OFF | ON |
| Tool parser | qwen3_coder | qwen3_xml | qwen3_coder |
| S+M1 (draft INT4) | ✅ | ❌ | ✅ |

## 2. Velocidad (client post-first tok/s, mediana n=5, 230W)

| Celda | **COMBO** | Sergio | Nuestro | Nota |
|---|---|---|---|---|
| p512/g128 | **117.1** (110-119) | 77.5 (68-78) | — | COMBO +51% vs Sergio |
| p8192/g128 | **105.7** (99-111) | 73.4 (65-74) | — | COMBO +44% vs Sergio |
| p512/g32 | 105.2 (93-106) | — | **117.3** (112-118) | breve brecha, ventana térmica |
| p8192/g32 | 99.3* (87-100) | — | **109.6** (106-111) | *final de sesión larga, MTP 0.78 colas |
| MTP aceptación | 0.88-0.98 (g128) | 0.89-0.93 | 0.93-0.98 | g32 colas con GPU caliente |

A g128 (coordenada de Sergio) el COMBO **iguala o supera nuestro récord previo** y
más que duplica la velocidad de su propia config para el MISMO modelo. La brecha a
g32 (105/99 vs 117/110) se explica por ventana térmica (+/-8-17% documentado) y por
diferencia de aceptación MTP en las colas de sesiones largas, no por la quant.

## 3. Robustez (gate exhaustivo — `exhaustive_test.py`)

| Módulo | **COMBO** | Sergio | Nuestro |
|---|---|---|---|
| A: 45 calidad | 45/45 | 45/45 | 45/45 |
| B: 40 secuencial | 40/40 | 40/40 | 40/40 |
| C: 8 concurrencia | **8/8** | **0/8 ✗ engine muerto** | 8/8 |
| D: 8K/32K | **OK** | **500 ✗** | OK |
| E: MTP ratio | 0.61 sano | 0.58 sano | 0.58 sano |
| **VEREDICTO** | **PASS** | **FAIL** | **PASS** |

El stack de Sergio (imagen 0.27.2 **sin** `patch_gdn_split_mixed.py`) muere con
2+ requests concurrentes o prefill largo mezclado con spec-decode:

```
RuntimeError: causal_conv1d does not support spec-decode and non-spec
(prefill + decode) tokens in the same invocation; ... mutually exclusive
```

Es el mismo bug que nosotros resolvimos con `patch_gdn_split_mixed.py` el 14-ago.
Su propio README lo admite ("use no-spec for concurrent serving"). En NUESTRO stack,
el MISMO modelo aguanta 8/8 y 8K/32K.

## 4. Calidad pura (evals limpios, misma seed/protocolo, think OFF)

| Test | **COMBO** | Sergio | Nuestro |
|---|---|---|---|
| **HumanEval pass@1** (164, ejecución real) | **92.7%** (152/164) | 92.1% (151/164) | 89.6% (147/164) |
| **MMLU** (238, 0-shot) | 72.7% (173/238) | **73.1%** (174/238) | 69.7% (166/238) |
| **MMLU-Pro** (140, 0-shot) | 62.9% (88/140) | 63.6% (89/140) | **65.7%** (92/140) |
| compile_fail HumanEval | 0 | 0 | 0 |

- La quant **gptqmodel de Sergio conserva algo más de calidad limpia** que nuestra
  AutoRound 512/4 (HumanEval +2.7, MMLU +3.4) — consistente con la jerarquía esperada
  del Plan B (`gptqmodel > AutoRound-pesado`). MMLU-Pro quedó en empate de ruido.
- El COMBO hereda esa calidad de su quant y, en HumanEval, marcó la mejor puntuación
  de la sesión (92.7).

## 5. PAGODA voxel (elaboración HTML, xhigh)

| Run | COMBO/Nuestro | Sergio (su stack) |
|---|---|---|
| multisample 16-ago (nuestro) | 168*, 66*, **162.201 chars** | — |
| 17-ago (2 runs c/u) | 31.748 · 33.903 · 1.420* | 30.097 · 23.868 |

(*) extracción/semilla parcial. **Conclusión: pura varianza de sampling**, no del
stack ni del quant — ambos caen típicamente en 23-34K chars, con outliers en 162K
(4× el "flagship" de Sergio, 39.6K). Detalle: `PAGODA-TEST.md`.

## 6. Decisión y ejecución

1. **Se ADOPTA el COMBO** como config del server (modelo gptqmodel + nuestro stack).
2. Modelo canónico: `Qwen3.8-27B-MTP-Preserved-GPTQ-Int4-gptqmodel` (copia local del repo de Sergio)
   (19G, 5 shards, 2399 tensores, 15 MTP BF16, GPTQ-INT4 g128 sym desc_act=False,
   lm_head BF16 — `quantize_config.json` + `quant_log.csv` verificado).
3. `launch_qwen38.sh` default → modelo gptqmodel. Fallbacks intactos:
   `Qwen3.8-27B-MTP-Preserved-GPTQ-Int4-heavy` (AutoRound 512/4) y ssd120 (256/2).
4. Nuestra quant (AutoRound 512/4) queda como **fallback de velocidad/robustez** y
   la línea 512/4 se descarta como primaria.
5. Los cambios propios (patches + esta comparativa) se publican en el fork `popisec`; el repositorio de Sergio queda intacto.

## 7. David vs Goliat — el stack de Sergio en su conjunto

Su imagen 0.27.2 + kernels 0.1.12.3 no compensan sin nuestros patches (S+M1,
gdn_split, ptr_wrap). Para reproducir SU config completa: `launch_sergio_qwen38.sh`
(sólo para A/B; no apta para producción por el crash de concurrencia).

## Datos crudos

- `runs/compare-sergio/` · `runs/compare-nosotros/` · `runs/compare-combo/`
  (exhaustive.log, evals/, cells/*/)
- Evals públicos: `evals/BENCHMARKS-PUBLICOS-20260816.md` (extendido 17-ago)

---

*Versión pública de la comparativa interna del proyecto, adaptada para el fork popisec* *(rutas de host y nombres de máquina eliminados). La matriz de mediciones regenerable está en `summary.json` (snapshot) y `tables.md`.*
