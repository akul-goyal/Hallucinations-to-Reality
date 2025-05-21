#!/usr/bin/env python3
"""
Script for analyzing large EDR log datasets with LLMs.

This script extends the standard analysis workflow to handle very large
datasets (tens of GB) by processing data in chunks and aggregating results.

Usage:
    python analyze_large_dataset.py --input PATH_TO_LARGE_FILE --output OUTPUT_DIR --config CONFIG_PATH
"""

import os
import sys
import json
import time
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.large_data_processor import LargeDataProcessor
from src.llm_analyzer import LLMAnalyzer
from src.visualizer import Visualizer
from src.utils import setup_logging, validate_config, save_results
import src.data_loader_patches
from src.data_loader import EDRDataLoader



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze large EDR logs using LLMs")
    
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the large EDR log file"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="./data/output", 
        help="Path to the output directory (default: ./data/output)"
    )
    
    parser.add_argument(
        "--config", "-c", 
        default="./config/config.yaml", 
        help="Path to the configuration file (default: ./config/config.yaml)"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int,
        default=500, 
        help="Size of each data chunk in MB (default: 500)"
    )
    
    parser.add_argument(
        "--provider", 
        choices=["anthropic", "deepseek", "deepseek_local"],
        help="Override the LLM provider in config"
    )
    
    parser.add_argument(
        "--model", 
        help="Override the model in config"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_false", 
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-api", "-n", 
        action="store_true", 
        help="Run without making API calls (for testing)"
    )
    
    parser.add_argument(
        "--local-model", "-l", 
        action="store_true", 
        help="Use local inference (requires local model setup)"
    )
    
    parser.add_argument(
        "--measure-perf", "-p", 
        action="store_true", 
        help="Measure and report performance metrics"
    )
    
    parser.add_argument(
        "--gpu-ids", 
        default="0", 
        help="Comma-separated list of GPU IDs to use (default: 0)"
    )

    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of input file even if already processed"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous analysis checkpoints"
    )
    
    return parser.parse_args()
    
    return parser.parse_args()

def load_config(config_path, args):
    """Load configuration from YAML file and apply command-line overrides."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validate configuration
        validate_config(config)
        
        # Apply command-line overrides
        if args.provider:
            config['llm']['provider'] = args.provider
        
        if args.model:
            config['llm']['model'] = args.model
        
        # Set local model if flag is provided
        if args.local_model:
            config['llm']['provider'] = "deepseek_local"
            
            # Ensure measure_performance is set if requested
            if args.measure_perf and 'deepseek_local' in config:
                config['deepseek_local']['measure_performance'] = True
        
        # Check for API key in environment variable if not in config
        if not config['llm']['api_key'] and not args.no_api:
            # Check for provider-specific environment variable
            provider = config['llm']['provider']
            env_var = {
                'anthropic': 'ANTHROPIC_API_KEY',
                'deepseek': 'DEEPSEEK_API_KEY'
            }.get(provider)
            
            if env_var:
                config['llm']['api_key'] = os.environ.get(env_var)
            
            if not config['llm']['api_key'] and provider != 'deepseek_local' and not args.no_api:
                raise ValueError(f"API key not found in config or environment variables ({env_var})")
        
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def setup_gpu_environment(gpu_ids):
    """Configure GPU environment."""
    # Set environment variables for CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            cuda_names = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
            logging.info(f"Found {cuda_count} CUDA devices: {cuda_names}")
            
            # Log memory stats for each device
            for i in range(cuda_count):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024**3)
                total_gb = total_mem / (1024**3)
                logging.info(f"GPU {i}: {free_gb:.2f}GB free of {total_gb:.2f}GB total")
        else:
            logging.warning("CUDA is not available. Running on CPU only.")
    except ImportError:
        logging.warning("PyTorch not available, skipping GPU detection.")

def main():
    """Main function for analyzing large datasets."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.CRITICAL
    setup_logging(log_level)
    
    # Configure GPU environment
    if args.local_model:
        setup_gpu_environment(args.gpu_ids)
    
    # Load configuration
    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config, args)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration used for this run
    config_output = output_dir / "run_config.yaml"
    with open(config_output, 'w') as f:
        yaml.dump(config, f)
    
    # Record start time for performance measurement
    start_time = time.time()
    
    try:
        # Initialize data loader with custom format support
        from src.data_loader import EDRDataLoader
        from src.data_loader_patches import apply_patches_to_data_loader
        
        data_loader = EDRDataLoader(config['edr'])
        data_loader = apply_patches_to_data_loader(data_loader)
        
        # Initialize large data processor with patched data loader
        logging.info(f"Initializing large data processor with chunk size {args.chunk_size}MB")
        processor = LargeDataProcessor(config['edr'], chunk_size_mb=args.chunk_size, data_loader=data_loader)
        
        # Process the large file into chunks
        logging.info(f"Processing large file: {args.input}")
        chunks_dir = processor.process_large_file(args.input, force_reprocess=args.force_reprocess)
        logging.info(f"Processed file into chunks at: {chunks_dir}")
        
        if not args.no_api:
            # Initialize LLM analyzer
            logging.info("Initializing LLM analyzer")
            analyzer = LLMAnalyzer(
                config['llm'],
                config['analysis'],
                config.get('deepseek'),  # Cloud Deepseek config
                config.get('deepseek_local')  # Local Deepseek config
            )
            
            # Analyze all chunks
            logging.info("Analyzing chunks with LLM")
            analysis_results = processor.analyze_chunks(
                analyzer, 
                output_dir, 
                resume=not args.no_resume
            )
        else:
            logging.info("Skipping LLM analysis (--no-api flag set)")
            analysis_results = {"mock_results": True, "timestamp": datetime.now().isoformat()}
        
        # Generate visualizations
        logging.info("Generating visualizations")
        visualizer = Visualizer(config['visualization'])
        
        # Create a sample DataFrame for the visualizer
        # Since we're not loading the entire dataset, just use the first chunk for visualizations
        sample_chunk = next(chunks_dir.glob("chunk_*.parquet"), None)
        if sample_chunk:
            sample_data = pd.read_parquet(sample_chunk)
            logging.info(f"Using {len(sample_data)} events from {sample_chunk.name} for visualizations")
        else:
            logging.warning("No chunks found for visualization. Creating empty DataFrame.")
            sample_data = pd.DataFrame()
        
        visualization_path = visualizer.create_visualizations(
            sample_data,
            analysis_results,
            output_dir
        )
        
        # Performance metrics
        elapsed_time = time.time() - start_time
        logging.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        # Add overall performance metrics
        if "performance" not in analysis_results:
            analysis_results["performance"] = {}
        
        analysis_results["performance"]["total_analysis_time"] = elapsed_time
        analysis_results["performance"]["input_file_size_gb"] = Path(args.input).stat().st_size / (1024**3)
        
        # Save final results
        final_results_path = save_results(analysis_results, output_dir)
        logging.info(f"Final results saved to {final_results_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        return 1
if __name__ == "__main__":
    sys.exit(main())
