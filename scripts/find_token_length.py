<invoke name="artifacts">
<parameter name="command">create</parameter>
<parameter name="id">gpu_max_context_estimator</parameter>
<parameter name="type">application/vnd.ant.code</parameter>
<parameter name="language">python</parameter>
<parameter name="title">gpu_max_context_estimator.py</parameter>
<parameter name="content">#!/usr/bin/env python3
"""
GPU Memory Estimator for vLLM Max Context Length

This script helps estimate the maximum context length (max-model-len) 
you can set for your model in vLLM based on your GPU's available memory.
"""

import argparse
import math
import subprocess
import json
import sys
import re
import os

def get_gpu_info():
    """Get information about available GPUs using nvidia-smi."""
    try:
        # Run nvidia-smi to get GPU information in JSON format
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        # Parse the CSV output
        for line in result.stdout.strip().split("\n"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                idx, name, total, free, used = parts
                
                # Extract memory values and convert to MB
                total_match = re.search(r"(\d+)\s*MiB", total)
                free_match = re.search(r"(\d+)\s*MiB", free)
                
                if total_match and free_match:
                    total_mb = int(total_match.group(1))
                    free_mb = int(free_match.group(1))
                    
                    gpus.append({
                        "index": idx,
                        "name": name,
                        "total_memory_mb": total_mb,
                        "free_memory_mb": free_mb,
                        "used_memory_mb": total_mb - free_mb
                    })
        
        return gpus
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: nvidia-smi command failed. Make sure you have NVIDIA drivers installed.")
        return []

def format_size(size_mb):
    """Format size in MB to a more readable format."""
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"

def estimate_max_context_length(model_name, gpu_memory_mb, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    """
    Estimate maximum context length based on GPU memory.
    
    Args:
        model_name: Name or size of the model (e.g., 7b, 13b, etc.)
        gpu_memory_mb: Available GPU memory in MB
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Target GPU memory utilization (0.0 to 1.0)
    
    Returns:
        Estimated maximum context length
    """
    # Extract model size if it's in the name
    model_size_match = re.search(r"(\d+\.?\d*)b", model_name.lower())
    model_size_billions = 7.0  # Default to 7B if not found
    
    if model_size_match:
        model_size_billions = float(model_size_match.group(1))
    
    # Effective memory after utilization factor
    effective_memory_mb = gpu_memory_mb * gpu_memory_utilization
    
    # Effective memory per model shard
    effective_memory_per_shard_mb = effective_memory_mb * tensor_parallel_size
    
    # Memory needed for model weights (very approximate)
    # For bfloat16/float16, each parameter needs 2 bytes
    model_weights_mb = (model_size_billions * 1000) / tensor_parallel_size
    
    # Available memory for activations and KV cache
    available_for_context_mb = effective_memory_per_shard_mb - model_weights_mb
    
    # Estimate KV cache size per token
    # This is a rough estimate and varies by model architecture
    # For a 7B model, each token in the context window uses ~128 bytes in KV cache
    bytes_per_token = (model_size_billions / 7.0) * 128
    
    # Maximum tokens based on available memory
    max_tokens = int((available_for_context_mb * 1024 * 1024) / bytes_per_token)
    
    # Ensure it's a multiple of 256 (common practice for efficiency)
    max_tokens = (max_tokens // 256) * 256
    
    return max(1024, min(max_tokens, 32768))  # Cap between 1024 and 32768

def main():
    parser = argparse.ArgumentParser(description="Estimate maximum context length for vLLM based on GPU memory")
    parser.add_argument("--model", default="mistral-7b-instruct", help="Model name or path")
    parser.add_argument("--gpu-util", type=float, default=0.9, help="Target GPU memory utilization (0.0-1.0)")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    # Get GPU information
    gpus = get_gpu_info()
    
    if not gpus:
        print("No GPUs detected or nvidia-smi not available.")
        sys.exit(1)
    
    print(f"Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Total Memory: {format_size(gpu['total_memory_mb'])}")
        print(f"    Free Memory: {format_size(gpu['free_memory_mb'])} ({gpu['free_memory_mb']/gpu['total_memory_mb']*100:.1f}%)")
    
    # Calculate for single GPU or multiple GPUs
    if args.tp_size > 1:
        if len(gpus) < args.tp_size:
            print(f"Warning: Requested tensor parallel size ({args.tp_size}) exceeds available GPUs ({len(gpus)})")
        
        # Sum the free memory across specified number of GPUs
        total_free_memory = sum(gpu["free_memory_mb"] for gpu in gpus[:args.tp_size])
        avg_free_memory = total_free_memory / args.tp_size
        
        print(f"\nTotal free memory across {args.tp_size} GPUs: {format_size(total_free_memory)}")
        print(f"Average free memory per GPU: {format_size(avg_free_memory)}")
        
        max_context = estimate_max_context_length(
            args.model, 
            avg_free_memory,  # Use average memory per GPU
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=args.gpu_util
        )
    else:
        # Use the GPU with the most free memory
        best_gpu = max(gpus, key=lambda g: g["free_memory_mb"])
        print(f"\nUsing GPU {best_gpu['index']} with {format_size(best_gpu['free_memory_mb'])} free memory")
        
        max_context = estimate_max_context_length(
            args.model,
            best_gpu["free_memory_mb"],
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_util
        )
    
    # Print recommendations
    print("\n=== RECOMMENDATIONS ===")
    
    # Conservative recommendation (safer)
    conservative = (max_context // 1024) * 1024
    if conservative <= 0:
        conservative = 1024
    
    # Moderate recommendation
    moderate = conservative
    if max_context >= 8192:
        moderate = 8192
    
    # Print the recommendations
    print(f"Model: {args.model}")
    print(f"GPU Memory Utilization: {args.gpu_util*100:.0f}%")
    print(f"Tensor Parallel Size: {args.tp_size} GPU(s)")
    
    print("\nEstimated maximum context lengths:")
    print(f"  Conservative: {conservative} tokens")
    print(f"  Moderate:     {moderate} tokens")
    print(f"  Aggressive:   {max_context} tokens")
    
    print("\nRecommended vLLM settings:")
    print(f"  --max-model-len {conservative}  # Conservative setting")
    
    # For more detailed information
    if args.verbose:
        print("\n=== DETAILED EXPLANATION ===")
        print("These estimates are approximate and depend on many factors including:")
        print("  - Exact model architecture and implementation")
        print("  - vLLM version and configuration")
        print("  - Other processes running on the GPU")
        print("  - System overhead")
        print("\nTo find the true maximum context length:")
        print("1. Start with the conservative estimate")
        print("2. Run a vLLM server with that setting")
        print("3. If stable, incrementally increase the value")
        print("4. Monitor GPU memory usage with 'nvidia-smi'")
        print("\nCommand to monitor GPU usage:")
        print('  watch -n 1 "nvidia-smi"')
        
        print("\nSample vLLM server command:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"    --model {args.model} \\")
        print(f"    --host 0.0.0.0 \\")
        print(f"    --port 8000 \\")
        print(f"    --tensor-parallel-size {args.tp_size} \\")
        print(f"    --gpu-memory-utilization {args.gpu_util} \\")
        print(f"    --max-model-len {conservative}")

if __name__ == "__main__":
    main()
</parameter>
</invoke>