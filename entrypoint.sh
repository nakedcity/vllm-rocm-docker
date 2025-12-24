#!/bin/bash
set -euo pipefail

echo "🔧 vLLM ROCm Entrypoint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Apply gfx1201 patch if needed
if [ "${IS_GFX1201:-0}" = "1" ]; then
    echo "🔧 Applying gfx1201 patch for aiter library..."
    echo "DEBUG: PYTHON_PATCH_SCRIPT='${PYTHON_PATCH_SCRIPT:-/patch_gfx1201.py}'"
    ls -la "${PYTHON_PATCH_SCRIPT:-/patch_gfx1201.py}" || echo "⚠️  File not found via ls"
    
    if [ -f "${PYTHON_PATCH_SCRIPT:-/patch_gfx1201.py}" ]; then
        python3 "${PYTHON_PATCH_SCRIPT:-/patch_gfx1201.py}" || {
            echo "⚠️  Warning: Failed to apply gfx1201 patch, continuing anyway..."
        }
    else
        echo "⚠️  Warning: Patch script not found, skipping patch"
    fi
fi

# Export critical ROCm environment variables
echo "🔧 Configuring ROCm environment..."
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-12.0.1}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

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
    --model "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}"
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-model-len "${MAX_MODEL_LEN:-8192}"
    --enforce-eager
)

# Add optional arguments
if [ -n "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}" ]; then
    VLLM_ARGS+=(--distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
fi

if [ -n "${QUANTIZATION:-}" ] && [ "$QUANTIZATION" != "none" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

if [ -n "${DTYPE:-}" ]; then
    VLLM_ARGS+=(--dtype "$DTYPE")
fi

# Execute vLLM server (replace shell process)
exec python3 -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
