#!/bin/bash
set -e

# User-configurable variables (set via command-line arguments)
MODEL=""
PORT=""
QUANTIZATION=""
DTYPE=""
GPU_MEMORY=""
MAX_MODEL_LEN_ARG=""
ARCHITECTURE=""
MULTI_GPU=0
LAUNCH_UI=0

# ROCm stability constants (hardcoded for reliability)
readonly HSA_ENABLE_SDMA_VAL=0
readonly NCCL_P2P_DISABLE_VAL=1
readonly NCCL_IB_DISABLE_VAL=1
readonly VLLM_ROCM_USE_AITER_VAL=0
readonly VLLM_ROCM_CUSTOM_PAGED_ATTN_VAL=0
readonly VLLM_ENABLE_V1_MULTIPROCESSING_VAL=0
readonly VLLM_USE_V1_VAL=0
readonly HIP_FORCE_DEV_KERNARG_VAL=1
readonly VLLM_ATTENTION_BACKEND_VAL="ROCM_ATTN"
readonly VLLM_DISTRIBUTED_EXECUTOR_BACKEND_VAL="uni"
readonly PYTORCH_ALLOC_CONF_VAL="expandable_segments:True"
readonly MAX_NUM_SEQS_VAL=64
readonly MAX_NUM_BATCHED_TOKENS_VAL=2048

# Argument parsing
show_help() {
    cat <<EOF
Usage: $0 [OPTIONS] [model_name]

ROCm-optimized vLLM server launcher for AMD GPUs

OPTIONS:
  --model MODEL           Hugging Face model ID or path
  --port PORT             Port to serve on (default: 9500)
  --quantization TYPE     Quantization: awq, gptq, none, or auto (default: auto)
  --dtype TYPE            Model dtype: auto, float16, bfloat16 (default: auto)
  --gpu-memory FRACTION   GPU memory utilization 0.0-1.0 (default: 0.55)
  --max-model-len LENGTH  Maximum context length (default: 8192)
  --architecture ARCH     GPU architecture preset (default: gfx1201)
                          Options: gfx1201, gfx1100, gfx1030, gfx90a
  --multi-gpu             Enable multi-GPU support (uses all available GPUs)
  --ui                    Launch Open WebUI alongside vLLM
  -h, --help              Show this help message

ARCHITECTURE PRESETS:
  gfx1201  RDNA 4 (Radeon AI PRO R9700)         HSA: 12.0.1
  gfx1100  RDNA 3 (RX 7000 series)              HSA: 11.0.0
  gfx1030  RDNA 2 (RX 6000 series)              HSA: 10.3.0
  gfx90a   CDNA 2 (MI200 series)                HSA: 9.0.10

EXAMPLES:
  $0                                             # Use defaults
  $0 mistralai/Mistral-7B-Instruct-v0.2         # Auto-detect quantization
  $0 --architecture gfx1100                      # Use RX 7000 series GPU
  $0 --gpu-memory 0.8 --max-model-len 16384     # High memory, long context
  $0 --multi-gpu --model meta-llama/Llama-2-70b # Multi-GPU for large model
  $0 --ui --port 8080                            # Launch with UI

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN_ARG="$2"
            shift 2
            ;;
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --multi-gpu)
            MULTI_GPU=1
            shift
            ;;
        --ui)
            LAUNCH_UI=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            else
                echo "❌ Unknown argument: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Apply defaults
MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
PORT="${PORT:-9500}"
DTYPE="${DTYPE:-auto}"
ARCHITECTURE="${ARCHITECTURE:-gfx1201}"
GPU_MEMORY="${GPU_MEMORY:-0.55}"
MAX_MODEL_LEN_ARG="${MAX_MODEL_LEN_ARG:-8192}"

# Auto-detect quantization from model name if not specified
if [ -z "$QUANTIZATION" ] || [ "$QUANTIZATION" = "auto" ]; then
    if [[ "$MODEL" =~ -AWQ ]]; then
        QUANTIZATION="awq"
        echo "🔍 Auto-detected quantization: AWQ (from model name)"
    elif [[ "$MODEL" =~ -GPTQ ]]; then
        QUANTIZATION="gptq"
        echo "🔍 Auto-detected quantization: GPTQ (from model name)"
    elif [[ "$MODEL" =~ -[Qq]uant ]]; then
        QUANTIZATION="awq"
        echo "🔍 Auto-detected quantization: AWQ (generic quantized model)"
    else
        QUANTIZATION="none"
        echo "🔍 Auto-detected quantization: none (base model)"
    fi
fi

# Warn if model name and quantization seem mismatched
if [ "$QUANTIZATION" = "awq" ] && [[ ! "$MODEL" =~ -AWQ ]] && [[ ! "$MODEL" =~ -[Qq]uant ]]; then
    echo "⚠️  Warning: Using AWQ quantization but model name doesn't contain '-AWQ'"
    echo "   Model: $MODEL"
    echo "   This may cause loading errors if the model is not actually AWQ-quantized"
fi

if [ "$QUANTIZATION" = "gptq" ] && [[ ! "$MODEL" =~ -GPTQ ]]; then
    echo "⚠️  Warning: Using GPTQ quantization but model name doesn't contain '-GPTQ'"
    echo "   Model: $MODEL"
    echo "   This may cause loading errors if the model is not actually GPTQ-quantized"
fi

if [ "$QUANTIZATION" = "none" ] && [[ "$MODEL" =~ -AWQ|-GPTQ|-[Qq]uant ]]; then
    echo "⚠️  Warning: Model name suggests quantization but --quantization is set to 'none'"
    echo "   Model: $MODEL"
    echo "   Consider using --quantization auto or specifying the correct type"
fi

# Export configuration (no environment variable fallbacks)
export MODEL
export PORT
export QUANTIZATION
export DTYPE
export GPU_MEMORY_UTILIZATION="$GPU_MEMORY"
export MAX_NUM_SEQS="$MAX_NUM_SEQS_VAL"
export MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS_VAL"
export MAX_MODEL_LEN="$MAX_MODEL_LEN_ARG"
export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF_VAL"

# Enable Triton AWQ optimization for ROCm when using AWQ quantization
if [ "$QUANTIZATION" = "awq" ]; then
    export VLLM_USE_TRITON_AWQ=1
fi

# Validate AMD GPU availability
if [ ! -e "/dev/kfd" ] || [ ! -e "/dev/dri" ]; then
    echo "❌ Error: AMD GPU devices not found (/dev/kfd or /dev/dri missing)"
    echo "   This script requires an AMD GPU with ROCm support."
    exit 1
fi

# Architecture-specific configuration
case "$ARCHITECTURE" in
    gfx1201)
        echo "🔧 Configuring for RDNA 4 (gfx1201) - Radeon AI PRO R9700"
        export IS_GFX1201=1
        export HSA_OVERRIDE_GFX_VERSION="12.0.1"
        export GPU_ARCHS="gfx1201"
        ;;
    gfx1100)
        echo "🔧 Configuring for RDNA 3 (gfx1100) - RX 7000 series"
        export IS_GFX1201=0
        export HSA_OVERRIDE_GFX_VERSION="11.0.0"
        export GPU_ARCHS="gfx1100"
        ;;
    gfx1030)
        echo "🔧 Configuring for RDNA 2 (gfx1030) - RX 6000 series"
        export IS_GFX1201=0
        export HSA_OVERRIDE_GFX_VERSION="10.3.0"
        export GPU_ARCHS="gfx1030"
        ;;
    gfx90a)
        echo "🔧 Configuring for CDNA 2 (gfx90a) - MI200 series"
        export IS_GFX1201=0
        export HSA_OVERRIDE_GFX_VERSION="9.0.10"
        export GPU_ARCHS="gfx90a"
        ;;
    *)
        echo "❌ Error: Unknown architecture '$ARCHITECTURE'"
        echo "   Supported: gfx1201, gfx1100, gfx1030, gfx90a"
        exit 1
        ;;
esac

# Multi-GPU configuration
if [ "$MULTI_GPU" = "1" ]; then
    echo "🔧 Enabling multi-GPU support"
    export HIP_VISIBLE_DEVICES=""  # Use all GPUs
else
    export HIP_VISIBLE_DEVICES=0   # Use first GPU only
fi

# Core ROCm environment variables
export VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND_VAL"
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND="$VLLM_DISTRIBUTED_EXECUTOR_BACKEND_VAL"

# ROCm stability settings
export HSA_ENABLE_SDMA="$HSA_ENABLE_SDMA_VAL"
export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE_VAL"
export NCCL_IB_DISABLE="$NCCL_IB_DISABLE_VAL"
export VLLM_ROCM_USE_AITER="$VLLM_ROCM_USE_AITER_VAL"
export VLLM_ROCM_CUSTOM_PAGED_ATTN="$VLLM_ROCM_CUSTOM_PAGED_ATTN_VAL"
export VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING_VAL"
export VLLM_USE_V1="$VLLM_USE_V1_VAL"
export HIP_FORCE_DEV_KERNARG="$HIP_FORCE_DEV_KERNARG_VAL"

# gfx1201 patch script (mounted into container)
export PYTHON_PATCH_SCRIPT="/patch_gfx1201.py"

# Display configuration
echo ""
echo "🚀 vLLM ROCm Server Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model:         $MODEL"
echo "Port:          $PORT"
echo "Quantization:  $QUANTIZATION"
echo "DType:         $DTYPE"
echo "GPU Memory:    ${GPU_MEMORY_UTILIZATION} (utilization)"
echo "Max Seqs:      $MAX_NUM_SEQS"
echo "UI Enabled:    $([ $LAUNCH_UI -eq 1 ] && echo 'Yes' || echo 'No')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Docker Compose profile options
if [ "$LAUNCH_UI" = "1" ]; then
    PROFILE_ARGS=(--profile ui)
else
    PROFILE_ARGS=()
fi

# Pull Docker images
echo "📦 Pulling Docker images..."
docker compose "${PROFILE_ARGS[@]}" pull

# Pre-download model
echo "📥 Downloading model '$MODEL'..."
docker compose run --rm vllm python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

# Start services
echo ""
echo "🔥 Starting vLLM server..."
echo "Press Ctrl+C to stop the services."
echo ""
docker compose "${PROFILE_ARGS[@]}" up

# Cleanup message
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 Services stopped."
echo "Run 'docker compose down' for full cleanup."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
