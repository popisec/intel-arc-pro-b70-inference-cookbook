#!/usr/bin/env bash
# B70 Qwen3.8-27B COMBO launcher — MTP4 dense serving, patched pinned nightly.
#
# WHEN TO USE: reproduce the adopted COMBO config on an Intel Arc Pro B70 —
#   the Sergio gptqmodel quant served on the 5-patch stack (concurrency + S+M1).
# WHAT IT DOES: applies ptr_wrap + gdn_split + draft S/M1 patches after the two
#   base nightly patches, then launches the pinned vLLM XPU nightly at 131328
#   context, fp8 KV, scheduler budget 4096, spec-decode MTP4 with tool calling.
# SAFETY: refuses a competing inference process; does not change host power.
#   ZE_AFFINITY_MASK defaults to 0 (single-GPU host); set ZE_AFFINITY_MASK=1 if
#   the B70 is the second GPU (e.g. next to an Arc A770).
#
# Example:
#   bash benchmarks/qwen38-27/launch-combo-mtp4-128k.sh /path/to/model 8000
set -euo pipefail

MODEL_DIR=${1:?usage: $0 MODEL_DIR [PORT]}
PORT=${2:-8000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE='vllm/vllm-openai-xpu@sha256:2c427ef477da092eb6f2cdbbbd24950b5fa171565b916db69d4c7bb10e68ca97'
MODEL_ID=${MODEL_ID:-$(basename "$MODEL_DIR")}
CONTAINER=b70-combo-qwen38

if pgrep -af 'llama-server|llama-bench|vllm serve|vllm-xpu' >/dev/null; then
  echo 'Refusing to launch: another inference process is active.' >&2
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

pkill -f "vllm serve" >/dev/null 2>&1 || true

# Order matters (mounted at entrypoint-defined paths):
#   /patch_mtp.py           -> patch_mtp_nightly.py
#   /patch_boundary.py      -> patch_mtp_boundary.py
#   /patch_ptr_wrap.py      -> patch_mtp_ptr_wrap.py
#   /patch_gdn_split.py     -> patch_gdn_split_mixed.py
#   /patch_loader_0.py      -> patch_draft_lmhead_int4.py
#   /patch_loader_1.py      -> patch_draft_mtp_int4.py
: "${ZE_AFFINITY_MASK:=0}"   # 0 = single-GPU host; 1 = B70 as GPU 1
RENDER_GID=$(stat -c '%g' /dev/dri/render* | sort -u | sed -n '1p')

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sudo docker run -d --name "$CONTAINER" -p "$PORT:8000" \
  --device /dev/dri --group-add "$RENDER_GID" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ROOT/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$ROOT/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v "$ROOT/patches/patch_mtp_ptr_wrap.py:/patch_ptr_wrap.py:ro" \
  -v "$ROOT/patches/patch_gdn_split_mixed.py:/patch_gdn_split.py:ro" \
  -v "$ROOT/patches/patch_draft_lmhead_int4.py:/patch_loader_0.py:ro" \
  -v "$ROOT/patches/patch_draft_mtp_int4.py:/patch_loader_1.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK="$ZE_AFFINITY_MASK" \
  -e B70_MTP_BF16_DRAFT=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e B70_DRAFT_LMHEAD_INT4=1 \
  -e B70_DRAFT_MTP_INT4=1 \
  -e B70_SPLIT_MIXED_GDN=1 \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; python /patch_ptr_wrap.py; python /patch_gdn_split.py; python /patch_loader_0.py; python /patch_loader_1.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 131328 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 64 --max-num-batched-tokens 4096 --enable-prefix-caching --served-model-name Qwen3.8-27B-MTP-Preserved-GPTQ-Int4 --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_coder --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'

printf 'Container started. Follow startup (first cold start ~15 min, graph capture):\n'
printf '  sudo docker logs -f %s\n' "$CONTAINER"
printf 'Health endpoint after startup:\n  curl -f http://127.0.0.1:%s/health\n' "$PORT"
printf 'OpenAI service (OK = MTP4 active):\n'
printf '  curl -s http://127.0.0.1:%s/v1/models\n' "$PORT"