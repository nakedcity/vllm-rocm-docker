#!/bin/bash
set -e

# Load defaults from template
if [ -f ".env-template" ]; then
    set -a
    source .env-template
    set +a
else
    echo "⚠️  Warning: .env-template not found. Using script hardcoded defaults as fallback."
fi

# Argument parsing logic
show_help() {
    cat <<EOF
Usage: $0 [OPTIONS] [model_name]

ROCm-optimized vLLM server launcher for AMD GPUs

OPTIONS:
  --model MODEL           Hugging Face model ID or path (Current: $MODEL)
  --port PORT             Port to serve on (Current: $PORT)
  --quantization TYPE     Quantization: awq, gptq, none, or auto (Current: $QUANTIZATION)
  --dtype TYPE            Model dtype: auto, float16, bfloat16 (Current: $DTYPE)
  --gpu-memory FRACTION   GPU memory utilization 0.0-1.0 (Current: $GPU_MEMORY_UTILIZATION)
  --max-model-len LENGTH  Maximum context length (Current: $MAX_MODEL_LEN)
  --tool-parser PARSER    Enable auto tool choice with parser (Current: $TOOL_CALL_PARSER)
                          Common: pythonic, hermes, mistral, llama3_json, granite
                          Pass empty string to disable.
  --reasoning-parser P    Reasoning parser to strip CoT tags (Current: $REASONING_PARSER)
                          Common: gemma4, openai_gptoss, deepseek_r1, qwen3, granite
                          Pass empty string to disable.
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

# Variable tracking for architecture logic
ARCHITECTURE="gfx1201" # Default architecture logic driver
MULTI_GPU=0
LAUNCH_UI=0
MODEL_ARG_SET=""

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
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --tool-parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        --reasoning-parser)
            REASONING_PARSER="$2"
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
            if [ "${1:0:2}" == "--" ]; then 
                echo "❌ Unknown argument: $1"
                show_help
                exit 1
            elif [ -z "$MODEL_ARG_SET" ]; then # Heuristic: First non-flag is model
                MODEL="$1"
                MODEL_ARG_SET=1
            else
                echo "❌ Unknown argument: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Auto-detect quantization from model name if not specified or auto
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

# Enable Triton AWQ optimization for ROCm when using AWQ quantization
if [ "$QUANTIZATION" = "awq" ]; then
    VLLM_USE_TRITON_AWQ=1
else
    VLLM_USE_TRITON_AWQ=0
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
        HSA_OVERRIDE_GFX_VERSION="12.0.1"
        GPU_ARCHS="gfx1201"
        ;;
    gfx1100)
        echo "🔧 Configuring for RDNA 3 (gfx1100) - RX 7000 series"
        HSA_OVERRIDE_GFX_VERSION="11.0.0"
        GPU_ARCHS="gfx1100"
        ;;
    gfx1030)
        echo "🔧 Configuring for RDNA 2 (gfx1030) - RX 6000 series"
        HSA_OVERRIDE_GFX_VERSION="10.3.0"
        GPU_ARCHS="gfx1030"
        ;;
    gfx90a)
        echo "🔧 Configuring for CDNA 2 (gfx90a) - MI200 series"
        HSA_OVERRIDE_GFX_VERSION="9.0.10"
        GPU_ARCHS="gfx90a"
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
    HIP_VISIBLE_DEVICES=""  # Use all GPUs
else
    HIP_VISIBLE_DEVICES=0   # Use first GPU only
fi

# Display configuration (Logging)
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

# Generate .env file with merged configuration
echo "📝 Generating .env file..."
cat > .env <<EOF
# Generated by run.sh on $(date)
# DO NOT EDIT THIS FILE DIRECTLY. Edit .env-template or pass arguments to run.sh

# Container Images
VLLM_IMAGE=$VLLM_IMAGE
OPEN_WEBUI_IMAGE=$OPEN_WEBUI_IMAGE

# Model Configuration
MODEL=$MODEL
PORT=$PORT
QUANTIZATION=$QUANTIZATION
DTYPE=$DTYPE
MAX_MODEL_LEN=$MAX_MODEL_LEN

# vLLM Performance
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF

# ROCm / Architecture
HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION
GPU_ARCHS=$GPU_ARCHS
HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA
NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE
NCCL_IB_DISABLE=$NCCL_IB_DISABLE
VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER
HIP_FORCE_DEV_KERNARG=$HIP_FORCE_DEV_KERNARG

# vLLM Engine
VLLM_USE_TRITON_AWQ=$VLLM_USE_TRITON_AWQ
TOOL_CALL_PARSER=$TOOL_CALL_PARSER
REASONING_PARSER=$REASONING_PARSER

# Hardware
HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES

# Secrets
HF_TOKEN=$HF_TOKEN
OPENAI_API_KEY=$OPENAI_API_KEY
EOF

# Docker Compose profile options
if [ "$LAUNCH_UI" = "1" ]; then
    PROFILE_ARGS=(--profile ui)
else
    PROFILE_ARGS=()
fi

# Cleanup previous containers to avoid name conflicts
echo "🧹 Cleaning up previous containers..."
docker compose down --remove-orphans >/dev/null 2>&1 || true
docker rm -f vllm-server >/dev/null 2>&1 || true
docker rm -f open-webui >/dev/null 2>&1 || true
# Stop Docker model runner to free GPU VRAM
docker stop docker-model-runner >/dev/null 2>&1 || true

# Pull Docker images
echo "📦 Pulling Docker images..."
docker compose "${PROFILE_ARGS[@]}" pull

# Pre-download model
# NOTE: override --entrypoint, otherwise the compose `entrypoint:` runs the
# custom vLLM launcher and ignores the python command (= boots a server here
# without published ports instead of downloading and exiting).
echo "📥 Downloading model '$MODEL'..."
if [ -n "$HF_TOKEN" ]; then
    docker compose run --rm --entrypoint python3 -e HF_TOKEN="$HF_TOKEN" vllm -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
else
    docker compose run --rm --entrypoint python3 vllm -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
fi

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
