# vLLM ROCm Server

ROCm-optimized vLLM server for AMD GPUs with automatic configuration and intelligent defaults.

## Quick Start

```bash
# Start with defaults
./run.sh

# Custom model
./run.sh mistralai/Mistral-7B-Instruct-v0.2

# With Web UI
./run.sh --ui
```

## Requirements

- Linux with Docker and Docker Compose
- AMD GPU with ROCm 6.2+
- Supported architectures: gfx1201 (RDNA 4), gfx1100 (RDNA 3), gfx1030 (RDNA 2), gfx90a (CDNA 2)

**Docker Images** (pinned versions for stability):
- vLLM: `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210` (ROCm 7.0.0, vLLM 0.11.2)
- Open WebUI: `ghcr.io/open-webui/open-webui:v0.4.5`

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | HuggingFace model ID |
| `--port` | `9500` | Server port |
| `--quantization` | `auto` | Quantization: `auto`, `awq`, `gptq`, `none` |
| `--dtype` | `auto` | Data type: `auto`, `float16`, `bfloat16` |
| `--gpu-memory` | `0.55` | GPU memory fraction (0.0-1.0) |
| `--max-model-len` | `8192` | Maximum context length |
| `--architecture` | `gfx1201` | GPU preset (see below) |
| `--multi-gpu` | disabled | Use all GPUs |
| `--ui` | disabled | Launch Open WebUI |

### GPU Architecture Presets

| Preset | GPU Family | Examples | HSA |
|--------|------------|----------|-----|
| `gfx1201` | RDNA 4 | Radeon AI PRO R9700 | 12.0.1 |
| `gfx1100` | RDNA 3 | RX 7900 XTX, RX 7800 XT | 11.0.0 |
| `gfx1030` | RDNA 2 | RX 6900 XT, RX 6800 XT | 10.3.0 |
| `gfx90a` | CDNA 2 | MI200 series | 9.0.10 |

## Examples

```bash
# RX 7000 series GPU
./run.sh --architecture gfx1100

# High memory + long context
./run.sh --gpu-memory 0.8 --max-model-len 16384

# Multi-GPU for large models
./run.sh --multi-gpu --model meta-llama/Llama-2-70b-chat-hf

# Custom settings
./run.sh --port 8080 --quantization none --dtype float16 --ui
```

## Features

### Auto-Detection
- **Quantization**: Automatically detects from model name (`-AWQ`, `-GPTQ`, etc.)
- **Warnings**: Alerts on model/quantization mismatches

### API Compatibility
OpenAI-compatible API at `http://localhost:9500/v1`:

```bash
# List models
curl http://localhost:9500/v1/models

# Generate completion
curl http://localhost:9500/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "prompt": "Hello!", "max_tokens": 50}'
```

## Troubleshooting

**Out of Memory**
```bash
./run.sh --gpu-memory 0.4
```

**GPU Not Detected**
```bash
ls -la /dev/kfd /dev/dri
rocm-smi
```

**Clean Restart**
```bash
docker compose down
docker system prune -f
./run.sh
```

## Architecture

- `run.sh` - Main launcher with ROCm configuration
- `entrypoint.sh` - Docker entrypoint with gfx1201 patching
- `patch_gfx1201.py` - Python script to patch aiter for gfx1201 support
- `docker-compose.yml` - Service definitions

## ROCm Optimizations

- HSA version auto-configured per GPU
- V1 engine disabled for stability
- Automatic aiter library patching for gfx1201
- Triton AWQ optimization enabled for AWQ models
- Optimized attention backend (ROCM_ATTN)
