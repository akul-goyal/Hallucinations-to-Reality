#!/usr/bin/env python3
"""
EDR Log Analysis with LLMs

This script processes Carbon Black EDR logs using Anthropic's Claude API
to detect and analyze potential security threats.

Usage:
    python main.py --input path/to/logs --output path/to/output --config path/to/config.yaml
"""

import os
import sys
import logging
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

# Add the parent directory to sys.path to enable module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import EDRDataLoader
from src.llm_analyzer import LLMAnalyzer
from src.visualizer import Visualizer
from src.utils import setup_logging, validate_config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze EDR logs using LLMs")
    
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to the input EDR log file or directory"
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
        "--verbose", "-v", 
        action="store_true", 
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
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validate configuration
        validate_config(config)
        
        # Check for API key in environment variable if not in config
        if not config['llm']['api_key']:
            config['llm']['api_key'] = os.environ.get('ANTHROPIC_API_KEY')
            if not config['llm']['api_key'] and not args.no_api:
                raise ValueError("API key not found in config or environment variables")
        
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record start time for performance measurement
    start_time = time.time()
    
    try:
        # Initialize components
        logging.info("Initializing data loader")
        data_loader = EDRDataLoader(config['edr'])
        
        logging.info("Loading EDR data")
        edr_data = data_loader.load(args.input)

        # Inside the main() function, modify the analyzer initialization
        if args.local_model:
            logging.info("Using local Deepseek model for inference")
            # Override provider in config if using local model flag
            config['llm']['provider'] = "deepseek_local"
            
            # Ensure measure_performance is set if requested
            if args.measure_perf and 'deepseek_local' in config:
                config['deepseek_local']['measure_performance'] = True

        
        if not args.no_api:
            logging.info("Initializing LLM analyzer")
            analyzer = LLMAnalyzer(
                config['llm'], 
                config['analysis'],
                config.get('deepseek'),  # Cloud Deepseek config
                config.get('deepseek_local')  # Local Deepseek config
            )
            
            logging.info("Analyzing EDR data with LLM")
            analysis_results = analyzer.analyze(edr_data)
        else:
            logging.info("Skipping LLM analysis (--no-api flag set)")
            analysis_results = {"mock_results": True, "timestamp": datetime.now().isoformat()}
        
        # Generate visualizations
        logging.info("Generating visualizations")
        visualizer = Visualizer(config['visualization'])
        visualization_path = visualizer.create_visualizations(
            edr_data, 
            analysis_results, 
            output_dir
        )
        
        # Performance metrics
        elapsed_time = time.time() - start_time
        event_count = len(edr_data) if hasattr(edr_data, "__len__") else "unknown"
        logging.info(f"Analysis completed in {elapsed_time:.2f} seconds for {event_count} events")
        
        # Output summary
        logging.info(f"Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())