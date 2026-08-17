#!/usr/bin/env bash
# B70 Qwen3.8-27B COMBO prefill/decode matrix — single-mode (MTP4) runner.
#
# WHEN TO USE: reproduce the phase-separated COMBO tables against a live
#   COMBO server (see launch-combo-mtp4-128k.sh). C1, n=5 per cell.
# WHAT IT DOES: generates exact entropy-first calibrated prompts inside the
#   server container, drives b70-realworld-context-harness.py through the 17
#   cells, then compiles summary.json + tables.md.
# OUTPUT: dumps/<run-id>/ with prompts/, per-cell results, manifest, summary.json.
#
# Example (server on :8000, container b70-combo-qwen38):
#   bash benchmarks/qwen38-27/b70-combo-prefill-decode-matrix.sh
# Env overrides: PORT (8000), CONTAINER (b70-combo-qwen38), MODEL_ID, RUN_OUT.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${PORT:-8000}
CONTAINER=${CONTAINER:-b70-combo-qwen38}
MODEL_ID=${MODEL_ID:-Qwen3.8-27B-MTP-Preserved-GPTQ-Int4}
RUN_ID="qwen38-combo-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_OUT=${RUN_OUT:-"$ROOT/results/$RUN_ID"}
SERVER="http://127.0.0.1:$PORT"

GEN=$ROOT/benchmarks/b70-generate-exact-prompts.py
HARNESS=$ROOT/benchmarks/b70-realworld-context-harness.py
COMPILE=$ROOT/benchmarks/qwen38-27/b70-combo-compile-tables.py
STD_SYS=$ROOT/benchmarks/benchmark-system-prompt.txt
PI_SYS=$ROOT/benchmarks/pi-system-prompt.txt

# name|prompt_tokens|requested_output|system|harness_flags
COORDS=(
  'prefill-p512|512|1|std|'
  'prefill-p2048|2048|1|std|'
  'prefill-p4096|4096|1|std|'
  'prefill-p6144|6144|1|std|'
  'prefill-p8192|8192|1|std|'
  'decode-p512-g32|512|32|std|--ignore-eos --full-output-warmup'
  'decode-p512-g128|512|128|std|--ignore-eos --full-output-warmup'
  'decode-p512-g256|512|256|std|--ignore-eos --full-output-warmup'
  'decode-p512-g512|512|512|std|--ignore-eos --full-output-warmup'
  'decode-p8192-g32|8192|32|std|--ignore-eos --full-output-warmup'
  'decode-p8192-g128|8192|128|std|--ignore-eos --full-output-warmup'
  'decode-control-p9445-g128|9445|128|std|--ignore-eos --full-output-warmup'
  'decode-p8192-g256|8192|256|std|--ignore-eos --full-output-warmup'
  'decode-p8192-g512|8192|512|std|--ignore-eos --full-output-warmup'
  'prefill-full-p131071|131071|1|pi|'
  'decode-full-p130944-g128|130944|128|pi|--ignore-eos --full-output-warmup'
  'decode-full-p130560-g512|130560|512|pi|--ignore-eos --full-output-warmup'
)
SELECT="${1:-ALL}"
WANT=()
if [ "$SELECT" = "ALL" ]; then WANT=("${COORDS[@]}"); else
  for c in "${COORDS[@]}"; do n="${c%%|*}"; [ "$n" = "$SELECT" ] && WANT+=("$c"); done
  [ "${#WANT[@]}" -gt 0 ] || { echo "unknown cell: $SELECT"; exit 2; }
fi

curl -sf "$SERVER/health" >/dev/null 2>&1 || { echo "server not up at $SERVER"; exit 1; }
mkdir -p "$RUN_OUT/prompts"

docker cp "$GEN" "$CONTAINER:/tmp/gen_prompts.py"
docker cp "$STD_SYS" "$CONTAINER:/tmp/system_std.txt"
docker cp "$PI_SYS" "$CONTAINER:/tmp/system_pi.txt"

for cell in "${WANT[@]}"; do
  IFS='|' read -r name tokens output sys flags <<< "$cell"
  sysfile="/tmp/system_pi.txt"; [ "$sys" = "std" ] && sysfile="/tmp/system_std.txt"
  docker exec "$CONTAINER" rm -f "/tmp/p_$name.json"
  docker exec "$CONTAINER" python /tmp/gen_prompts.py \
    --model /model --system-prompt-file "$sysfile" \
    --output "/tmp/p_$name.json" --targets "$tokens" --per-target 6
  docker cp "$CONTAINER:/tmp/p_$name.json" "$RUN_OUT/prompts/$name.json"
  docker exec "$CONTAINER" rm -f "/tmp/p_$name.json"
done

for cell in "${WANT[@]}"; do
  IFS='|' read -r name tokens output sys flags <<< "$cell"
  if [ "$output" -eq 1 ]; then exact=""; else exact="--ignore-eos --full-output-warmup"; fi
  python3 "$HARNESS" --mode context \
    --prompts "$RUN_OUT/prompts/$name.json" --target "$tokens" \
    --output "$output" --budget 4096 --reps 5 --model "$MODEL_ID" \
    --root "$SERVER" --outdir "$RUN_OUT/$name" $exact \
    > "$RUN_OUT/$name.harness.log" 2>&1 || {
      echo "cell FAILED: $name (see $RUN_OUT/$name.harness.log)"; exit 3; }
done

{
  echo "run_id=$RUN_ID"
  echo 'registry_agreement=b70_combo_v1'
  echo "image=vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97"
  echo "model_id=$MODEL_ID"
  echo 'patches=patch_mtp_nightly.py,patch_mtp_boundary.py,patch_mtp_ptr_wrap.py,patch_gdn_split_mixed.py,patch_draft_lmhead_int4.py,patch_draft_mtp_int4.py'
  echo 'mode=mtp4 num_speculative_tokens=4 scheduler_budget=4096 max_model_len=131328 kv_cache=fp8 gpu_util=0.88 cap_W=230'
  echo 'prefix_cache=on cold_start_entropy_first zero_hit_delta'
  echo 'n_per_cell=5 warmup_fraction=1 timing_source=client_monotonic_SSE'
  echo "server=$SERVER container=$CONTAINER"
} > "$RUN_OUT/manifest.txt"

python3 "$COMPILE" "$RUN_OUT"
echo "compiled: $RUN_OUT/tables.md"