#!/bin/bash
# Script to start vLLM server with the correct model path
# Usage: ./start_vllm_server.sh [model_path] [port]

MODEL_PATH=${1:-"../models/mistral-7b-instruct"}
PORT=${2:-8000}

# Convert to absolute path if needed
if [[ "$MODEL_PATH" == /* ]]; then
    # Already absolute path
    ABSOLUTE_MODEL_PATH="$MODEL_PATH"
elif [[ "$MODEL_PATH" == ./* ]] || [[ "$MODEL_PATH" == ../* ]]; then
    # Relative path, convert to absolute
    ABSOLUTE_MODEL_PATH="$(realpath "$MODEL_PATH")"
else
    # Hugging Face model ID, use as is
    ABSOLUTE_MODEL_PATH="$MODEL_PATH"
fi

echo "Starting vLLM server with model: $ABSOLUTE_MODEL_PATH"

# Kill any existing vLLM server
pkill -f "vllm.entrypoints.openai.api_server" || echo "No existing server found"

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model "$ABSOLUTE_MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096

exit $?
