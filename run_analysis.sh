#!/bin/bash
# Script to run the EDR log analysis on a GPU server
# Usage: ./run_analysis.sh [path_to_logs] [output_dir]

set -e  # Exit on error

# Default values
LOG_PATH=${1:-"/path/to/your/80gb/edr_logs.json"}
OUTPUT_DIR=${2:-"./results"}
CONFIG_PATH="./config/large_datasets_config.yaml"
GPU_IDS="0"  # Use first GPU by default
CHUNK_SIZE=500  # MB per chunk

# Check if we're running on a system with NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Setting up environment..."
    nvidia-smi
    # Get available GPU memory
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    
    echo "Total GPU memory: ${TOTAL_MEM}MB, Free memory: ${FREE_MEM}MB"
    
    # If we have multiple GPUs, use them all
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if [ $NUM_GPUS -gt 1 ]; then
        # Make sure we're using an even number of GPUs for tensor parallelism
        # This is needed because the model's attention heads must be divisible by GPU count
        EVEN_GPUS=$(( $NUM_GPUS - ($NUM_GPUS % 2) ))
        
        # If we have an odd number, use one fewer GPU
        if [ $EVEN_GPUS -ne $NUM_GPUS ]; then
            echo "Detected $NUM_GPUS GPUs, but using $EVEN_GPUS for compatibility with model architecture"
        fi
	NUM_GPUS=$EVEN_GPUS
        
        # Generate comma-separated list of GPU IDs (0,1,2,3...)
        GPU_IDS=$(seq -s, 0 $(($EVEN_GPUS-1)))
        echo "Using multiple GPUs: $GPU_IDS"
    fi
else
    echo "No NVIDIA GPUs detected. Using CPU mode."
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Start the vLLM server (if using local model)
start_vllm_server() {
    echo "Starting vLLM server for local inference..."
    
    # Convert model path to absolute path if needed
    if [[ "$model" == /* ]]; then
        # Already absolute path
        ABSOLUTE_MODEL_PATH="$model"
    elif [[ "$model" == ./* ]] || [[ "$model" == ../* ]]; then
        # Relative path, convert to absolute
        ABSOLUTE_MODEL_PATH="$(realpath "$model")"
    else
        # Hugging Face model ID, use as is
        ABSOLUTE_MODEL_PATH="$model"
    fi
    
    echo "Using model: $ABSOLUTE_MODEL_PATH"
    
     python -m vllm.entrypoints.openai.api_server \
        --model "$ABSOLUTE_MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $([ $NUM_GPUS -gt 1 ] && echo $NUM_GPUS || echo 1) \
        --gpu-memory-utilization 0.9 \
        --max-model-len 32768 &
    
    # Wait for server to start
    echo "Waiting for vLLM server to start..."
    sleep 30
    
    # Check if server is running
    curl -s http://localhost:8000/v1/models > /dev/null
    if [ $? -eq 0 ]; then
        echo "vLLM server is running"
    else
        echo "vLLM server failed to start"
        exit 1
    fi
}

# Main analysis function
run_analysis() {
    echo "Starting analysis of EDR logs: $LOG_PATH"
    echo "Output directory: $OUTPUT_DIR"
    echo "Configuration: $CONFIG_PATH"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run the analysis
    python scripts/analyze_large_datasets.py \
        --input "$LOG_PATH" \
        --output "$OUTPUT_DIR" \
        --config "$CONFIG_PATH" \
        --gpu-ids "$GPU_IDS" \
        --chunk-size "$CHUNK_SIZE" \
        --local-model \
        --measure-perf \
        --verbose
    
    RESULT=$?
    
    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Analysis completed in $(( DURATION / 3600 ))h $(( (DURATION % 3600) / 60 ))m $(( DURATION % 60 ))s"
    
    # Check if analysis was successful
    if [ $RESULT -eq 0 ]; then
        echo "Analysis completed successfully"
    else
        echo "Analysis failed with exit code $RESULT"
        exit $RESULT
    fi
    # Run benchmarking
    echo "Running benchmark analysis on results..."
    python scripts/benchmark_results.py \
        --results "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/benchmark" \
        --verbose

    if [ $? -eq 0 ]; then
        echo "Benchmarking completed successfully"
        echo "Benchmark report available at: $OUTPUT_DIR/benchmark"
    else
        echo "Benchmarking failed"
    fi
}

# Check if log file exists
if [ ! -f "$LOG_PATH" ]; then
    echo "Error: Log file not found: $LOG_PATH"
    exit 1
fi

# Get file size in GB
FILE_SIZE=$(du -BG "$LOG_PATH" | cut -f1 | tr -d 'G')
echo "Log file size: ${FILE_SIZE}GB"

# Check if we need to start the local model server
# Find the section where the model is determined, typically near this code:
if grep -q "provider[\"]*: [\"']*deepseek_local" "$CONFIG_PATH"; then
    echo "Local model configuration detected"
    
    # Extract the model path specifically from the deepseek_local section
    MODEL_PATH=$(awk '/deepseek_local:/,/^[^ ]/ {if ($1 == "model:") print $2}' "$CONFIG_PATH" | tr -d '",' | tr -d "'")
    
    # If MODEL_PATH is empty, try a different approach
    if [ -z "$MODEL_PATH" ]; then
        echo "Attempting alternative extraction method..."
        MODEL_PATH=$(grep -A5 "deepseek_local:" "$CONFIG_PATH" | grep "model:" | head -1 | awk '{print $2}' | tr -d '",' | tr -d "'")
    fi
    
    # Provide a default if still empty
    if [ -z "$MODEL_PATH" ]; then
        echo "Warning: Could not extract model path from config, using default."
        MODEL_PATH="./models/mistral-7b-instruct"
    fi
    
    echo "Found model path in config: $MODEL_PATH"
    
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
    
    echo "Using model path: $ABSOLUTE_MODEL_PATH"
    
    # Update the model variable for passing to start_vllm_server
    model="$ABSOLUTE_MODEL_PATH"
    
    start_vllm_server
fi

# Increase max open files limit if needed for large files
ulimit -n 65535 2>/dev/null || echo "Could not increase file limit (not running as root)"

# Set environment variables for performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Run the analysis
run_analysis

echo "Analysis process complete."
echo "Results are available in: $OUTPUT_DIR"
echo "To view the report, open: $OUTPUT_DIR/benchmark/benchmark_report_*.md"

# Stop the vLLM server if it was started
if grep -q "provider[\"]*: [\"']*deepseek_local" "$CONFIG_PATH"; then
    echo "Stopping vLLM server..."
    pkill -f "vllm.entrypoints.openai.api_server" || echo "Server already stopped"
fi

exit 0
