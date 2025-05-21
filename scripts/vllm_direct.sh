#!/bin/bash
# Kill any existing vLLM server
pkill -f "vllm.entrypoints.openai.api_server"

# Path to Mistral model
MODEL_PATH="/home/akulg2/edr-llm-analyzer/models/mistral-7b-instruct"

echo "Starting vLLM server with Mistral 7B Instruct"

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096

exit $?