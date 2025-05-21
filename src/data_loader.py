"""
Data Loader for EDR Logs

This module handles loading and preprocessing of EDR logs from various sources,
with a focus on Carbon Black.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any

try:
    from cbapi.response import CbResponseAPI, Process
    CBAPI_AVAILABLE = True
except ImportError:
    CBAPI_AVAILABLE = False
    logging.warning("cbapi not available. Direct Carbon Black API access will be disabled.")

class EDRDataLoader:
    """
    Class for loading and preprocessing EDR logs from different sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Dictionary containing EDR configuration parameters
        """
        self.config = config
        self.source = config.get("source", "carbon_black")
        self.log_format = config.get("log_format", "json")
        self.time_window = config.get("time_window", 24)  # hours
        self.max_events = config.get("max_events", 1000)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Carbon Black API client if needed and available
        self.cb_api = None
        if self.source == "carbon_black" and CBAPI_AVAILABLE:
            try:
                self.cb_api = CbResponseAPI()
                self.logger.info("Successfully connected to Carbon Black API")
            except Exception as e:
                self.logger.error(f"Failed to initialize Carbon Black API: {e}")
    
    def load(self, input_path: str) -> pd.DataFrame:
        """
        Load EDR data from the specified source.
        
        Args:
            input_path: Path to the input file or directory, or query for API
            
        Returns:
            DataFrame containing processed EDR events
        """
        self.logger.info(f"Loading EDR data from {input_path}")
        
        if self.source == "carbon_black":
            if os.path.exists(input_path):
                return self._load_from_file(input_path)
            elif self.cb_api is not None:
                return self._load_from_cb_api(input_path)
            else:
                raise ValueError("Carbon Black API not available and input path does not exist")
        else:
            raise ValueError(f"Unsupported EDR source: {self.source}")
    
    def _load_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load EDR data from a file.
        
        Args:
            file_path: Path to the file containing EDR data
            
        Returns:
            DataFrame containing processed EDR events
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if path.is_dir():
            return self._load_from_directory(path)
        
        self.logger.info(f"Loading data from file: {file_path}")
        
        if self.log_format == "json":
            return self._load_json_file(path)
        elif self.log_format == "csv":
            return self._load_csv_file(path)
        else:
            raise ValueError(f"Unsupported log format: {self.log_format}")
    
    def _load_from_directory(self, directory: Path) -> pd.DataFrame:
        """
        Load EDR data from all files in a directory.
        
        Args:
            directory: Path to the directory containing EDR data files
            
        Returns:
            DataFrame containing processed EDR events from all files
        """
        self.logger.info(f"Loading data from directory: {directory}")
        
        dataframes = []
        
        if self.log_format == "json":
            for file_path in directory.glob("*.json"):
                dataframes.append(self._load_json_file(file_path))
        elif self.log_format == "csv":
            for file_path in directory.glob("*.csv"):
                dataframes.append(self._load_csv_file(file_path))
        else:
            raise ValueError(f"Unsupported log format: {self.log_format}")
        
        if not dataframes:
            raise ValueError(f"No {self.log_format} files found in {directory}")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Sort by timestamp
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp")
        
        # Limit the number of events if needed
        if len(combined_df) > self.max_events:
            self.logger.warning(
                f"Limiting events to {self.max_events} (from {len(combined_df)} total)"
            )
            combined_df = combined_df.iloc[:self.max_events]
        
        return combined_df
    
    def _load_json_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load EDR data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            DataFrame containing processed EDR events
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "events" in data:
                df = pd.DataFrame(data["events"])
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame([data])
            
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {e}")
            raise
    
    def load_csv_file(file_path: Path) -> pd.DataFrame:
        """
        Load network flow data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing processed network flow events
        """
        try:
            # Load CSV with custom column names since there are no headers
            df = pd.read_csv(file_path, 
                             header=None, 
                             names=["source", "destination", "action"])
            
            # Add additional columns needed for analysis
            df["event_id"] = [f"evt-{i+1}" for i in range(len(df))]
            df["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df), freq='s')
            
            return df
                
        except Exception as e:
            logging.error(f"Error loading CSV file {file_path}: {e}")
            raise
    
    def _load_from_cb_api(self, query: str) -> pd.DataFrame:
        """
        Load EDR data directly from Carbon Black API.
        
        Args:
            query: Query string for Carbon Black API
            
        Returns:
            DataFrame containing processed EDR events
        """
        if not CBAPI_AVAILABLE or self.cb_api is None:
            raise RuntimeError("Carbon Black API is not available")
        
        self.logger.info(f"Querying Carbon Black API with: {query}")
        
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.time_window)
        
        try:
            # Query processes
            processes = self.cb_api.select(Process).where(query)
            processes = processes.min_last_update(start_time).max_last_update(end_time)
            
            # Limit the results
            processes = processes[:self.max_events]
            
            # Convert to list of dictionaries
            events = []
            for process in processes:
                process_dict = {
                    "id": process.id,
                    "process_name": process.process_name,
                    "process_pid": process.process_pid,
                    "hostname": process.hostname,
                    "username": process.username,
                    "command_line": process.cmdline,
                    "path": process.path,
                    "parent_id": process.parent_id,
                    "parent_name": process.parent_name,
                    "start_time": process.start or "",
                    "last_update": process.last_update or "",
                    "process_md5": process.process_md5,
                    "webui_link": process.webui_link
                }
                
                # Add events
                for event in process.events:
                    event_dict = process_dict.copy()
                    event_dict.update({
                        "event_type": event.get("event_type", ""),
                        "timestamp": event.get("timestamp", ""),
                        "event_data": json.dumps(event)
                    })
                    events.append(event_dict)
            
            # Convert to DataFrame
            df = pd.DataFrame(events)
            return self._preprocess_dataframe(df)
            
        except Exception as e:
            self.logger.error(f"Error querying Carbon Black API: {e}")
            raise
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame containing EDR events.
        
        Args:
            df: DataFrame with raw EDR events
            
        Returns:
            Preprocessed DataFrame
        """
        # Check if DataFrame is empty
        if df.empty:
            self.logger.warning("Empty dataset loaded")
            return df
        
        # Standardize column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # Ensure required columns
        required_columns = ["timestamp", "event_type"]
        for col in required_columns:
            if col not in df.columns:
                if col == "timestamp" and "time" in df.columns:
                    df["timestamp"] = df["time"]
                elif col == "event_type" and "type" in df.columns:
                    df["event_type"] = df["type"]
                else:
                    self.logger.warning(f"Required column '{col}' not found in data")
        
        # Convert timestamp to datetime if available
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                # Sort by timestamp
                df = df.sort_values("timestamp")
                # Drop rows with invalid timestamps
                df = df.dropna(subset=["timestamp"])
            except Exception as e:
                self.logger.warning(f"Error converting timestamps: {e}")
        
        # Add severity if not present
        if "severity" not in df.columns:
            # Default severity logic based on event type
            severity_mapping = {
                "process": 3,
                "netconn": 4,
                "filemod": 3,
                "regmod": 3,
                "moduleload": 2,
                "childproc": 3,
                "crossproc": 5,
                "remotethread": 7,
                "tamper": 8,
                "blocked": 7,
                "alert": 8
                # Add more mappings as needed
            }
            
            if "event_type" in df.columns:
                df["severity"] = df["event_type"].map(
                    lambda x: severity_mapping.get(str(x).lower(), 1)
                )
            else:
                df["severity"] = 1  # Default severity
        
        # Add unique event ID if not present
        if "event_id" not in df.columns:
            df["event_id"] = [f"evt-{i}" for i in range(len(df))]
        
        # Limit to max events
        if len(df) > self.max_events:
            self.logger.warning(
                f"Limiting events to {self.max_events} (from {len(df)} total)"
            )
            df = df.iloc[:self.max_events]
        
        return df