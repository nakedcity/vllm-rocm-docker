#!/bin/bash
set -euo pipefail

echo "🔧 vLLM ROCm Entrypoint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Export critical ROCm environment variables
echo "🔧 Configuring ROCm environment..."
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-12.0.1}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"

# Display environment for debugging
echo "🔍 ROCm Environment Variables:"
env | grep -E '^(VLLM|HSA|HIP|ROCM|NCCL)' | sort || true
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Validate required variables
if [ -z "${MODEL:-}" ]; then
    echo "❌ Error: MODEL environment variable is not set"
    exit 1
fi

if [ -z "${PORT:-}" ]; then
    echo "❌ Error: PORT environment variable is not set"
    exit 1
fi

# Build vLLM server arguments
echo "🚀 Starting vLLM Server..."
echo "   Model: $MODEL"
echo "   Port: $PORT"
echo "   GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.55}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VLLM_ARGS=(
    "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-model-len "${MAX_MODEL_LEN:-8192}"
    --enforce-eager
)

if [ -n "${QUANTIZATION:-}" ] && [ "$QUANTIZATION" != "none" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [ -n "${DTYPE:-}" ]; then
    VLLM_ARGS+=(--dtype "$DTYPE")
fi

if [ -n "${TOOL_CALL_PARSER:-}" ]; then
    echo "🛠  Enabling auto tool choice (parser: $TOOL_CALL_PARSER)"
    VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

# Execute vLLM server (replace shell process)
exec vllm serve "${VLLM_ARGS[@]}"
