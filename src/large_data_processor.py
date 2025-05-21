"""
Large Data Processor for EDR Log Analysis

This module extends the data loader to handle very large datasets (tens of GB)
using streaming and chunking techniques.
"""

import os
import json
import logging
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Tuple, Union
from tqdm import tqdm

from src.data_loader import EDRDataLoader

import pickle
import hashlib
from datetime import datetime



class LargeDataProcessor:
    """
    Class for processing and analyzing large EDR log datasets.
    """
    
    def __init__(self, config: Dict[str, Any], chunk_size_mb: int = 500, data_loader=None):
        """
        Initialize the large data processor with configuration.
        
        Args:
            config: Dictionary containing EDR configuration parameters
            chunk_size_mb: Size of each data chunk in MB for processing
            data_loader: Optional custom data loader instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader
        if self.data_loader is None:
            from src.data_loader import EDRDataLoader
            self.data_loader = EDRDataLoader(config)
        self.chunk_size_mb = chunk_size_mb
        
        # Create directory for processed chunks
        self.chunks_dir = Path("data/processed_chunks")
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Add checkpointing directories
        self.checkpoints_dir = Path("data/checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Add state tracking
        self.state = {
            "file_hash": None,
            "file_path": None,
            "chunks_processed": 0,
            "chunks_analyzed": 0,
            "file_size": 0,
            "last_updated": None,
            "config_hash": self._hash_config(config),
            "chunk_paths": []
        }
        
        # Load previous state if available
        self._load_state()

    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash of the config to track config changes."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _hash_file(self, file_path: Path) -> str:
        """Create a hash of the file to track file changes."""
        # For very large files, just hash the first 1MB + file size + modification time
        file_stat = file_path.stat()
        size = file_stat.st_size
        mtime = file_stat.st_mtime
        
        # Read the first MB
        with open(file_path, 'rb') as f:
            content = f.read(1024 * 1024)
        
        # Create hash
        hasher = hashlib.md5()
        hasher.update(content)
        hasher.update(str(size).encode())
        hasher.update(str(mtime).encode())
        
        return hasher.hexdigest()

    def _save_state(self):
        """Save the current processing state."""
        state_path = self.checkpoints_dir / "processor_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(self.state, f)
        
        self.logger.info(f"Saved processing state to {state_path}")

    def _load_state(self):
        """Load the previous processing state if available."""
        state_path = self.checkpoints_dir / "processor_state.pkl"
        if state_path.exists():
            try:
                with open(state_path, 'rb') as f:
                    self.state = pickle.load(f)
                self.logger.info(f"Loaded previous processing state from {state_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load state, starting fresh: {e}")

    def _handle_complex_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle complex data types that aren't compatible with Parquet.
        Converts dictionaries and lists containing dictionaries to JSON strings.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame with complex types converted to strings
        """
        for col in df.columns:
            # Skip if column is entirely None or already string type
            if df[col].isna().all() or pd.api.types.is_string_dtype(df[col]):
                continue
                
            try:
                # Check for dict/map types
                if df[col].apply(lambda x: isinstance(x, dict)).any():
                    # Convert dictionaries to strings
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
                
                # Check for list types containing dictionaries
                elif df[col].apply(lambda x: isinstance(x, list) and any(isinstance(item, dict) for item in x if isinstance(x, list))).any():
                    # Convert lists with dictionaries to strings
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) and any(isinstance(item, dict) for item in x) else x)
                
                # Special handling for flattened data
                elif col == 'flattened_data':
                    # Ensure flattened_data is always a string
                    df[col] = df[col].astype(str)
                    
            except Exception as e:
                # If anything goes wrong, convert the entire column to strings as a fallback
                self.logger.warning(f"Error handling complex types in column {col}: {e}")
                df[col] = df[col].astype(str)
                    
        return df
        
    def process_large_file(self, file_path: str, force_reprocess: bool = False) -> Path:
        """
        Process a large EDR log file by splitting it into manageable chunks.
        
        Args:
            file_path: Path to the large EDR log file
            force_reprocess: If True, reprocess even if already processed
            
        Returns:
            Path to the directory containing processed chunks
        """
        self.logger.info(f"Processing large file: {file_path}")
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if this file has already been processed
        file_hash = self._hash_file(file_path)
        file_size = file_path.stat().st_size
        
        if (not force_reprocess and 
            self.state["file_hash"] == file_hash and 
            self.state["file_size"] == file_size and
            self.state["file_path"] == str(file_path) and
            self.state["chunk_paths"]):
            
            self.logger.info(f"File already processed, using cached chunks")
            # Verify chunks exist
            all_chunks_exist = all(Path(path).exists() for path in self.state["chunk_paths"])
            
            if all_chunks_exist:
                return self.chunks_dir
            else:
                self.logger.warning("Some chunk files are missing, reprocessing")
        
        # Update state for new file
        self.state["file_hash"] = file_hash
        self.state["file_path"] = str(file_path)
        self.state["file_size"] = file_size
        self.state["chunks_processed"] = 0
        self.state["chunk_paths"] = []
        self.state["last_updated"] = datetime.now().isoformat()
        
        # Save state before processing
        self._save_state()
        
        # Check file format and process
        if file_path.suffix.lower() == '.json':
            return self._process_large_json(file_path, force_reprocess)
        elif file_path.suffix.lower() == '.csv':
            return self._process_large_csv(file_path, force_reprocess)
        elif file_path.suffix.lower() in ['.parquet', '.pq']:
            return self._process_large_parquet(file_path, force_reprocess)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _process_large_json(self, file_path: Path, force_reprocess: bool = False) -> Path:
        """
        Process a large JSON file by reading it in chunks.
        
        Args:
            file_path: Path to the JSON file
            force_reprocess: If True, reprocess even if already processed
            
        Returns:
            Path to the directory containing processed chunks
        """
        self.logger.info(f"Processing large JSON file: {file_path}")
        
        if force_reprocess:
            # Clean up previous chunks
            for chunk_file in self.chunks_dir.glob("chunk_*.parquet"):
                chunk_file.unlink()
        
        # Process file in chunks to avoid loading everything into memory
        chunk_paths = []
        
        # Estimate the JSON format first by reading a small portion
        with open(file_path, 'r') as f:
            start_bytes = f.read(10000)  # Read first 10KB to determine format
        
        # Check if the JSON starts with an array or an object with an events key
        is_array = start_bytes.strip().startswith('[')
        is_events_obj = '"events"' in start_bytes and start_bytes.strip().startswith('{')
        
        # Process based on detected format
        if is_array:
            self.logger.info("Detected JSON array format")
            chunk_paths = self._process_json_array(file_path, force_reprocess)
        elif is_events_obj:
            self.logger.info("Detected JSON object with 'events' key")
            chunk_paths = self._process_json_events_object(file_path, force_reprocess)
        else:
            self.logger.info("Detected JSON lines or non-standard format, using line-by-line processing")
            chunk_paths = self._process_json_lines(file_path, force_reprocess)
        
        self.logger.info(f"Processed JSON file into {len(chunk_paths)} chunks")
        return self.chunks_dir
    
    def _process_json_array(self, file_path: Path, force_reprocess: bool = False) -> List[Path]:
        """Process a large JSON file in array format."""
        import ijson  # For iterative JSON parsing
        
        # Check if we can skip processing or resume
        if not force_reprocess and self.state["chunks_processed"] > 0 and self.state["file_path"] == str(file_path):
            self.logger.info(f"Resuming from {self.state['chunks_processed']} previously processed chunks")
            # Verify chunk files exist
            all_exist = True
            for chunk_path in self.state["chunk_paths"]:
                if not Path(chunk_path).exists():
                    all_exist = False
                    break
            
            if all_exist:
                return [Path(path) for path in self.state["chunk_paths"]]
        
        # Create checkpoint frequency
        checkpoint_frequency = 5  # Save state every 5 chunks
        
        chunk_paths = []
        chunk_data = []
        total_size = 0
        chunk_count = 0
        
        with open(file_path, 'rb') as f:
            # Iterate through JSON array items
            for item in tqdm(ijson.items(f, 'item'), desc="Processing JSON array"):
                # Process and flatten the item
                processed_item = self._process_and_flatten_item(item)
                chunk_data.append(processed_item)
                
                # Rough size estimation (use the flattened string for size estimation)
                item_size = len(json.dumps(processed_item).encode('utf-8'))
                total_size += item_size
                
                # If chunk size exceeds threshold, save and start a new chunk
                if total_size >= self.chunk_size_mb * 1024 * 1024:
                    chunk_count += 1
                    chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
                    
                    # Convert to DataFrame and save as Parquet
                    df = pd.DataFrame(chunk_data)
                    df = self._preprocess_dataframe(df)
                    df = self._handle_complex_types(df)
                    df.to_parquet(chunk_path)
                    
                    # Update state
                    chunk_paths.append(chunk_path)
                    self.state["chunks_processed"] = chunk_count
                    self.state["chunk_paths"] = [str(p) for p in chunk_paths]
                    self.state["last_updated"] = datetime.now().isoformat()
                    
                    # Save checkpoint periodically
                    if chunk_count % checkpoint_frequency == 0:
                        self._save_state()
                        self.logger.info(f"Saved checkpoint after processing {chunk_count} chunks")
                    
                    chunk_data = []
                    total_size = 0
        
        # Save any remaining data
        if chunk_data:
            chunk_count += 1
            chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
            df = pd.DataFrame(chunk_data)
            df = self._preprocess_dataframe(df)
            df = self._handle_complex_types(df)
            df.to_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        
        # Update final state
        self.state["chunks_processed"] = chunk_count
        self.state["chunk_paths"] = [str(p) for p in chunk_paths]
        self.state["last_updated"] = datetime.now().isoformat()
        self._save_state()
        
        return chunk_paths
    
    def _process_json_events_object(self, file_path: Path, force_reprocess: bool = False) -> List[Path]:
        """Process a large JSON file with an 'events' array key."""
        import ijson  # For iterative JSON parsing
        
        # Check if we can skip processing or resume
        if not force_reprocess and self.state["chunks_processed"] > 0 and self.state["file_path"] == str(file_path):
            self.logger.info(f"Resuming from {self.state['chunks_processed']} previously processed chunks")
            # Verify chunk files exist
            all_exist = True
            for chunk_path in self.state["chunk_paths"]:
                if not Path(chunk_path).exists():
                    all_exist = False
                    break
            
            if all_exist:
                self.logger.info("All chunk files found, using cached results")
                return [Path(path) for path in self.state["chunk_paths"]]
        
        # Create checkpoint frequency
        checkpoint_frequency = 5  # Save state every 5 chunks
        
        chunk_paths = []
        chunk_data = []
        total_size = 0
        chunk_count = 0
        
        with open(file_path, 'rb') as f:
            # Iterate through events array items
            for item in tqdm(ijson.items(f, 'events.item'), desc="Processing JSON events"):
                # Process and flatten the item
                processed_item = self._process_and_flatten_item(item)
                chunk_data.append(processed_item)
                
                # Rough size estimation
                item_size = len(json.dumps(processed_item).encode('utf-8'))
                total_size += item_size
                
                # If chunk size exceeds threshold, save and start a new chunk
                if total_size >= self.chunk_size_mb * 1024 * 1024:
                    chunk_count += 1
                    chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
                    
                    # Convert to DataFrame and save as Parquet
                    df = pd.DataFrame(chunk_data)
                    df = self._preprocess_dataframe(df)
                    df = self._handle_complex_types(df)
                    df.to_parquet(chunk_path)
                    
                    # Update state
                    chunk_paths.append(chunk_path)
                    self.state["chunks_processed"] = chunk_count
                    self.state["chunk_paths"] = [str(p) for p in chunk_paths]
                    self.state["last_updated"] = datetime.now().isoformat()
                    
                    # Save checkpoint periodically
                    if chunk_count % checkpoint_frequency == 0:
                        self._save_state()
                        self.logger.info(f"Saved checkpoint after processing {chunk_count} chunks")
                    
                    chunk_data = []
                    total_size = 0
        
        # Save any remaining data
        if chunk_data:
            chunk_count += 1
            chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
            df = pd.DataFrame(chunk_data)
            df = self._preprocess_dataframe(df)
            df = self._handle_complex_types(df)
            df.to_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        
        # Update final state
        self.state["chunks_processed"] = chunk_count
        self.state["chunk_paths"] = [str(p) for p in chunk_paths]
        self.state["last_updated"] = datetime.now().isoformat()
        self._save_state()
        
        return chunk_paths
    
    def _process_json_lines(self, file_path: Path, force_reprocess: bool = False) -> List[Path]:
        """Process a large JSON file one line at a time (JSON Lines format)."""
        chunk_paths = []
        chunk_data = []
        total_size = 0
        chunk_count = 0
        line_count = 0
        
        # Check if we can skip processing or resume from a checkpoint
        if not force_reprocess and self.state["chunks_processed"] > 0 and self.state["file_path"] == str(file_path):
            self.logger.info(f"Resuming from {self.state['chunks_processed']} previously processed chunks")
            # Verify chunk files exist
            all_exist = True
            for chunk_path in self.state["chunk_paths"]:
                if not Path(chunk_path).exists():
                    all_exist = False
                    break
            
            if all_exist:
                self.logger.info("All chunk files found, skipping processing")
                return [Path(path) for path in self.state["chunk_paths"]]
            else:
                self.logger.warning("Some chunks missing, will reprocess file")
                # Reset state for this file
                self.state["chunks_processed"] = 0
                self.state["chunk_paths"] = []
        
        # Clean up previous chunks if needed
        if force_reprocess or self.state["chunks_processed"] == 0:
            for chunk_file in self.chunks_dir.glob("chunk_*.parquet"):
                chunk_file.unlink()
        
        # Create checkpoint frequency
        checkpoint_frequency = 5  # Save state every 5 chunks
        
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(file_path, 'r'))
        
        with open(file_path, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Processing JSON lines"):
                line = line.strip()
                if not line or line in ['[', ']', '{', '}', '[{', '}]']:  # Skip brackets
                    continue
                
                # Remove trailing commas if present
                if line.endswith(','):
                    line = line[:-1]
                
                try:
                    item = json.loads(line)
                    # Process and flatten the item
                    processed_item = self._process_and_flatten_item(item)
                    chunk_data.append(processed_item)
                    
                    # Rough size estimation
                    total_size += len(line.encode('utf-8'))
                    line_count += 1
                    
                    # If chunk size exceeds threshold, save and start a new chunk
                    if total_size >= self.chunk_size_mb * 1024 * 1024:
                        chunk_count += 1
                        chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(chunk_data)
                        
                        # Preprocess and handle complex types
                        df = self._preprocess_dataframe(df)
                        df = self._handle_complex_types(df)
                        
                        # Save as Parquet
                        df.to_parquet(chunk_path)
                        
                        # Update state
                        chunk_paths.append(chunk_path)
                        self.state["chunks_processed"] = chunk_count
                        self.state["chunk_paths"] = [str(p) for p in chunk_paths]
                        self.state["last_updated"] = datetime.now().isoformat()
                        
                        # Save checkpoint periodically
                        if chunk_count % checkpoint_frequency == 0:
                            self._save_state()
                            self.logger.info(f"Saved checkpoint after processing {chunk_count} chunks")
                        
                        chunk_data = []
                        total_size = 0
                except json.JSONDecodeError:
                    self.logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
        
        # Save any remaining data
        if chunk_data:
            chunk_count += 1
            chunk_path = self.chunks_dir / f"chunk_{chunk_count:04d}.parquet"
            df = pd.DataFrame(chunk_data)
            df = self._preprocess_dataframe(df)
            df = self._handle_complex_types(df)
            df.to_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        
        # Update final state
        self.state["chunks_processed"] = chunk_count
        self.state["chunk_paths"] = [str(p) for p in chunk_paths]
        self.state["last_updated"] = datetime.now().isoformat()
        self._save_state()
        
        self.logger.info(f"Processed {line_count} JSON lines into {chunk_count} chunks")
        return chunk_paths
  
    def _process_and_flatten_item(self, item):
        """
        Process a JSON item and convert it to a flattened format with escaped special characters.
        
        Args:
            item: JSON item to process
            
        Returns:
            Dictionary with flattened representation of the item
        """
        if isinstance(item, dict):
            try:
                # Add both the original JSON (for backward compatibility) and the flattened version
                flattened_str = self._flatten_json_object(item)
                return {
                    "original_json": json.dumps(item),  # Store as JSON string to maintain original structure
                    "flattened_data": flattened_str
                }
            except Exception as e:
                # If flattening fails, log the error and return a simple representation
                self.logger.warning(f"Error flattening JSON object: {e}. Using simple string representation.")
                return {
                    "original_json": json.dumps(item),
                    "flattened_data": f"Error processing: {str(e)[:100]}...",
                    "processing_error": str(e)
                }
        else:
            # If it's not a dictionary, keep as is but ensure it's safely escaped
            safe_value = self._escape_special_chars(str(item)) if item is not None else "null"
            return {"raw_value": safe_value}

    
    def _flatten_json_object(self, json_obj: dict, prefix: str = '') -> str:
        """
        Flatten a JSON object into a single string with properly escaped special characters.
        
        Args:
            json_obj: The JSON object to flatten
            prefix: Prefix for nested keys (used in recursion)
            
        Returns:
            A flattened string representation of the JSON object with escaped special characters
        """
        flattened_items = []
        
        for key, value in json_obj.items():
            # Escape special characters in keys
            safe_key = self._escape_special_chars(str(key))
            
            # Create the current key with prefix if it exists
            current_key = f"{prefix}::{safe_key}" if prefix else safe_key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened_items.append(self._flatten_json_object(value, current_key))
            elif isinstance(value, list):
                # Handle lists by joining their string representations
                if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                    # Simple list of primitives - escape each value
                    list_items = []
                    for item in value:
                        if item is None:
                            list_items.append("null")
                        else:
                            list_items.append(self._escape_special_chars(item))
                    
                    list_str = ", ".join(list_items)
                    flattened_items.append(f"{current_key}: [{list_str}]")
                else:
                    # Complex list with dictionaries or nested structures
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            # Recursively flatten dictionary items in the list
                            flattened_items.append(self._flatten_json_object(item, f"{current_key}[{i}]"))
                        else:
                            # Handle primitive values in the list
                            if item is None:
                                item_str = "null"
                            else:
                                item_str = self._escape_special_chars(item)
                            flattened_items.append(f"{current_key}[{i}]: {item_str}")
            else:
                # Handle primitive values - convert None to "null" and escape others
                if value is None:
                    value_str = "null"
                else:
                    value_str = self._escape_special_chars(value)
                
                flattened_items.append(f"{current_key}: {value_str}")
        
        # Join all flattened items with commas
        return ", ".join(flattened_items)

    def _escape_special_chars(self, text: str) -> str:
        """
        Escape special characters in a string to make it safe for JSON and HTTP requests.
        
        Args:
            text: The string to escape
            
        Returns:
            Escaped string that is safe for JSON and HTTP requests
        """
        if not isinstance(text, str):
            return str(text)
        
        # Use json.dumps to properly escape the string for JSON
        # This handles quotes, backslashes, control characters, etc.
        escaped = json.dumps(text)[1:-1]  # Remove the surrounding quotes that json.dumps adds
        
        # Additional escaping for characters that might cause issues in HTTP requests
        http_problematic = {
            # These can sometimes cause issues in specific HTTP contexts
            "<": "\\u003C",  # Less than sign (to avoid HTML/XML injection)
            ">": "\\u003E",  # Greater than sign
            "&": "\\u0026",  # Ampersand
        }
        
        for char, replacement in http_problematic.items():
            escaped = escaped.replace(char, replacement)
        
        return escaped
    
    def _process_large_csv(self, file_path: Path) -> Path:
        """
        Process a large CSV file using Dask for out-of-memory computation.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Path to the directory containing processed chunks
        """
        self.logger.info(f"Processing large CSV file: {file_path}")
        
        # Clean up previous chunks
        for chunk_file in self.chunks_dir.glob("chunk_*.parquet"):
            chunk_file.unlink()
        
        # Read CSV file with Dask
        ddf = dd.read_csv(
            file_path,
            blocksize=self.chunk_size_mb * 1024 * 1024,  # Block size in bytes
            assume_missing=True,  # Handle missing values
            sample=10000,  # Sample rows to infer schema
            low_memory=True
        )
        
        # Process and save each partition as a Parquet file
        chunk_paths = []
        for i, partition in enumerate(tqdm(ddf.partitions, desc="Processing CSV partitions")):
            chunk_path = self.chunks_dir / f"chunk_{i+1:04d}.parquet"
            
            # Convert partition to pandas and preprocess
            df = partition.compute()
            df = self._preprocess_dataframe(df)
            
            # Save as Parquet
            df.to_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        
        self.logger.info(f"Processed CSV file into {len(chunk_paths)} chunks")
        return self.chunks_dir
    
    def _process_large_parquet(self, file_path: Path) -> Path:
        """
        Process a large Parquet file by splitting it into smaller chunks.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            Path to the directory containing processed chunks
        """
        self.logger.info(f"Processing large Parquet file: {file_path}")
        
        # Clean up previous chunks
        for chunk_file in self.chunks_dir.glob("chunk_*.parquet"):
            chunk_file.unlink()
        
        # Read Parquet file with PyArrow
        parquet_file = pq.ParquetFile(file_path)
        chunk_paths = []
        
        # Get number of row groups
        num_row_groups = parquet_file.num_row_groups
        self.logger.info(f"Parquet file has {num_row_groups} row groups")
        
        # Process each row group
        for i in tqdm(range(num_row_groups), desc="Processing Parquet row groups"):
            chunk_path = self.chunks_dir / f"chunk_{i+1:04d}.parquet"
            
            # Read row group
            table = parquet_file.read_row_group(i)
            df = table.to_pandas()
            
            # Preprocess and save
            df = self._preprocess_dataframe(df)
            df.to_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        
        self.logger.info(f"Processed Parquet file into {len(chunk_paths)} chunks")
        return self.chunks_dir
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a DataFrame of EDR events.
        
        Args:
            df: DataFrame with raw EDR events
            
        Returns:
            Preprocessed DataFrame
        """
        # Use the data loader's preprocessing function
        return self.data_loader._preprocess_dataframe(df)
    
    def analyze_chunks(self, analyzer, output_dir: Path, resume: bool = True) -> Dict[str, Any]:
        """
        Analyze all processed chunks using the provided analyzer.
        
        Args:
            analyzer: LLM analyzer instance to use
            output_dir: Directory to save results
            resume: If True, attempt to resume from previous analysis
            
        Returns:
            Dictionary containing aggregated analysis results
        """
        self.logger.info("Analyzing processed chunks")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all chunk files
        chunk_files = sorted(list(self.chunks_dir.glob("chunk_*.parquet")))
        if not chunk_files:
            raise ValueError("No chunks found to analyze")
        
        self.logger.info(f"Found {len(chunk_files)} chunks to analyze")
        
        # Check for existing results to resume
        chunk_results = []
        start_index = 0
        
        if resume:
            # Check if we can resume from previous analysis
            existing_results = list(output_dir.glob("chunk_result_*.json"))
            if existing_results:
                existing_indices = [int(p.stem.split('_')[-1]) for p in existing_results]
                max_index = max(existing_indices) if existing_indices else 0
                
                # Load existing results
                for i in range(1, max_index + 1):
                    result_path = output_dir / f"chunk_result_{i}.json"
                    if result_path.exists():
                        try:
                            with open(result_path, 'r') as f:
                                chunk_results.append(json.load(f))
                                self.logger.info(f"Loaded existing result for chunk {i}")
                        except Exception as e:
                            self.logger.warning(f"Failed to load result for chunk {i}: {e}")
                            # Stop at first failure to ensure sequential analysis
                            break
                
                # Set starting point for analysis
                start_index = len(chunk_results)
                self.logger.info(f"Resuming analysis from chunk {start_index + 1}")
        
        # Analyze remaining chunks
        for i, chunk_file in enumerate(tqdm(chunk_files[start_index:], 
                                          initial=start_index, 
                                          total=len(chunk_files), 
                                          desc="Analyzing chunks")):
            chunk_index = start_index + i
            self.logger.info(f"Analyzing chunk {chunk_index+1}/{len(chunk_files)}: {chunk_file}")
            
            # Load chunk
            df = pd.read_parquet(chunk_file)
            
            # Skip empty chunks
            if df.empty:
                self.logger.warning(f"Skipping empty chunk: {chunk_file}")
                continue
            
            # Analyze chunk
            try:
                result = analyzer.analyze(df)
                
                # Save individual chunk result
                chunk_result_path = output_dir / f"chunk_result_{chunk_index+1}.json"
                with open(chunk_result_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                chunk_results.append(result)
                
                # Update state
                self.state["chunks_analyzed"] = chunk_index + 1
                self.state["last_updated"] = datetime.now().isoformat()
                self._save_state()
                
                self.logger.info(f"Saved chunk {chunk_index+1} result to {chunk_result_path}")
            except Exception as e:
                self.logger.error(f"Error analyzing chunk {chunk_index+1}: {e}", exc_info=True)
                # Save state even on error to allow resuming after fixing the error
                self._save_state()
        
        # Aggregate results from all chunks
        aggregated_results = self._aggregate_results(chunk_results)
        
        # Save aggregated results
        aggregated_path = output_dir / "aggregated_results.json"
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        self.logger.info(f"Saved aggregated results to {aggregated_path}")
        return aggregated_results
    
    def _aggregate_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple chunks.
        
        Args:
            chunk_results: List of dictionaries containing chunk results
            
        Returns:
            Dictionary containing aggregated results
        """
        self.logger.info(f"Aggregating results from {len(chunk_results)} chunks")
        
        # Combine threats from all chunks
        all_threats = []
        all_patterns = []
        all_recommendations = set()
        all_limitations = set()
        all_attack_chains = []
        
        # Track unique threats and patterns to avoid duplicates
        seen_threat_ids = set()
        seen_pattern_ids = set()
        
        # Collect results from each chunk
        for i, result in enumerate(chunk_results):
            chunk_id = f"chunk_{i+1}"
            
            # Process threats
            for threat in result.get("identified_threats", []):
                # Generate a unique ID if needed
                if "threat_id" not in threat or not threat["threat_id"]:
                    threat["threat_id"] = f"threat-{len(seen_threat_ids)+1}"
                
                # Skip if duplicate
                if threat["threat_id"] in seen_threat_ids:
                    continue
                
                # Add chunk info
                threat["source_chunk"] = chunk_id
                seen_threat_ids.add(threat["threat_id"])
                all_threats.append(threat)
            
            # Process patterns
            for pattern in result.get("suspicious_patterns", []):
                # Generate a unique ID if needed
                if "pattern_id" not in pattern or not pattern["pattern_id"]:
                    pattern["pattern_id"] = f"pattern-{len(seen_pattern_ids)+1}"
                
                # Skip if duplicate
                if pattern["pattern_id"] in seen_pattern_ids:
                    continue
                
                # Add chunk info
                pattern["source_chunk"] = chunk_id
                seen_pattern_ids.add(pattern["pattern_id"])
                all_patterns.append(pattern)
            
            # Collect recommendations and limitations
            for rec in result.get("recommendations", []):
                all_recommendations.add(rec)
                
            for lim in result.get("limitations", []):
                all_limitations.add(lim)
            
            # Collect attack chains
            for chain in result.get("potential_attack_chains", []):
                chain["source_chunk"] = chunk_id
                all_attack_chains.append(chain)
        
        # Assess overall risk
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for threat in all_threats:
            severity = threat.get("severity", "Low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Determine overall risk level
        if severity_counts["High"] > 0:
            risk_level = "High"
        elif severity_counts["Medium"] > 0:
            risk_level = "Medium"
        elif severity_counts["Low"] > 0:
            risk_level = "Low"
        else:
            risk_level = "Minimal"
        
        risk_assessment = {
            "overall_risk_level": risk_level,
            "reasoning": (
                f"Based on {len(all_threats)} identified threats across {len(chunk_results)} data chunks "
                f"({severity_counts['High']} high, {severity_counts['Medium']} medium, "
                f"{severity_counts['Low']} low severity)."
            ),
            "threat_count": len(all_threats),
            "severity_distribution": severity_counts
        }
        
        # Calculate performance metrics
        total_api_calls = sum(result.get("performance", {}).get("api_calls", 0) for result in chunk_results)
        total_tokens = sum(result.get("performance", {}).get("total_tokens", 0) for result in chunk_results)
        total_duration = sum(result.get("performance", {}).get("duration_seconds", 0) for result in chunk_results)
        
        performance = {
            "total_api_calls": total_api_calls,
            "total_tokens": total_tokens,
            "total_duration_seconds": total_duration,
            "chunks_processed": len(chunk_results),
            "average_tokens_per_chunk": total_tokens / max(1, len(chunk_results)),
            "average_duration_per_chunk": total_duration / max(1, len(chunk_results))
        }
        
        # Create final summary
        final_summary = {
            "conclusion": f"Analysis of {len(chunk_results)} chunks identified {len(all_threats)} potential threats.",
            "confidence": "Medium",
            "attack_summary": self._generate_attack_summary(all_threats, all_attack_chains),
            "key_indicators": self._extract_key_indicators(all_threats, all_patterns),
            "limitations": list(all_limitations)
        }
        
        return {
            "identified_threats": all_threats,
            "suspicious_patterns": all_patterns,
            "risk_assessment": risk_assessment,
            "recommendations": list(all_recommendations),
            "limitations": list(all_limitations),
            "potential_attack_chains": all_attack_chains,
            "performance": performance,
            "final_summary": final_summary,
            "chunk_count": len(chunk_results),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    def _generate_attack_summary(self, threats: List[Dict[str, Any]], 
                                attack_chains: List[Dict[str, Any]]) -> str:
        """Generate a summary of the attack based on threats and attack chains."""
        # Check if there are high severity threats
        high_threats = [t for t in threats if t.get("severity") == "High"]
        
        if not threats:
            return "No significant threats identified in the analyzed data."
        
        if not high_threats and not attack_chains:
            return f"Analysis identified {len(threats)} potential threats but no clear attack patterns were established."
        
        # Get the most significant attack chain
        significant_chain = None
        if attack_chains:
            # Sort chains by severity and number of threats
            sorted_chains = sorted(
                attack_chains,
                key=lambda c: (
                    1 if c.get("severity") == "High" else (0.5 if c.get("severity") == "Medium" else 0),
                    len(c.get("threats", []))
                ),
                reverse=True
            )
            if sorted_chains:
                significant_chain = sorted_chains[0]
        
        if significant_chain:
            chain_severity = significant_chain.get("severity", "Medium")
            threat_count = len(significant_chain.get("threats", []))
            chain_description = self._summarize_chain(significant_chain)
            
            return (
                f"Analysis identified a {chain_severity.lower()} severity attack chain involving {threat_count} "
                f"connected threats. {chain_description}"
            )
        else:
            # Summarize the most severe threats
            threat_types = list(set(t.get("threat_type", "Unknown") for t in high_threats[:3]))
            return (
                f"Analysis identified {len(high_threats)} high severity threats including "
                f"{', '.join(threat_types)}. However, no clear attack chain was established."
            )
    
    def _summarize_chain(self, chain: Dict[str, Any]) -> str:
        """Generate a summary of an attack chain."""
        threats = chain.get("threats", [])
        if not threats:
            return "No details available for this chain."
        
        # Group threats by type
        threat_types = {}
        for threat in threats:
            t_type = threat.get("threat_type", "Unknown")
            if t_type not in threat_types:
                threat_types[t_type] = 0
            threat_types[t_type] += 1
        
        # Create a summary of the chain
        type_summary = ", ".join(f"{count} {t_type}" for t_type, count in threat_types.items())
        
        # Try to determine the attack sequence
        first_threat = threats[0] if threats else {}
        last_threat = threats[-1] if threats else {}
        
        first_type = first_threat.get("threat_type", "Unknown activity")
        last_type = last_threat.get("threat_type", "Unknown activity")
        
        return f"The attack chain involves {type_summary}, beginning with {first_type} and progressing to {last_type}."
    
    def _extract_key_indicators(self, threats: List[Dict[str, Any]], 
                              patterns: List[Dict[str, Any]]) -> List[str]:
        """Extract key indicators of compromise from threats and patterns."""
        indicators = []
        
        # Extract from threats
        for threat in threats:
            if threat.get("severity") == "High":
                desc = threat.get("description", "")
                if desc and len(desc) > 10:  # Only include meaningful descriptions
                    # Truncate and clean up the description
                    indicator = desc.split(".")[0]  # Take first sentence
                    if len(indicator) > 100:
                        indicator = indicator[:97] + "..."
                    indicators.append(indicator)
        
        # Extract from patterns
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "")
            if pattern_type and pattern_type not in indicators:
                indicators.append(pattern_type)
        
        # Limit to top 5 indicators
        return sorted(set(indicators))[:5]
