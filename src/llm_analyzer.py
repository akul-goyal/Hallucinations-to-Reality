"""
LLM Analyzer for EDR Logs

This module handles the interaction with LLM APIs (Anthropic Claude or Deepseek)
to analyze EDR logs and identify potential security threats.
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# For resource monitoring with local models
try:
    import psutil
    import torch
    import gc
    import os
    RESOURCE_MONITORING_AVAILABLE = True
    
    # Try importing GPU monitoring tools
    try:
        import GPUtil
        GPU_MONITORING_AVAILABLE = True
    except ImportError:
        GPU_MONITORING_AVAILABLE = False
        logging.warning("GPUtil not available. GPU monitoring will be limited.")
        
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    logging.warning("Resource monitoring packages not available. Performance tracking will be limited.")


# Import LLM API clients
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic package not available. Claude API access will be disabled.")

try:
    import openai
    OPENAI_AVAILABLE = True  # For Deepseek which uses OpenAI-compatible API
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. Deepseek API access will be disabled.")

class LLMAnalyzer:
    """
    Class for analyzing EDR logs using LLMs.
    """
    
    def __init__(self, llm_config: Dict[str, Any], analysis_config: Dict[str, Any], 
            deepseek_config: Optional[Dict[str, Any]] = None,
            deepseek_local_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM analyzer with configuration.
        
        Args:
            llm_config: Dictionary containing LLM configuration parameters
            analysis_config: Dictionary containing analysis configuration parameters
            deepseek_config: Optional dictionary containing Deepseek-specific configuration
        """
        self.llm_config = llm_config
        self.analysis_config = analysis_config
        self.deepseek_config = deepseek_config
        self.deepseek_local_config = deepseek_local_config
        self.logger = logging.getLogger(__name__)

        self.resource_usage = []
        self.measure_performance = False
        
        # Determine provider
        self.provider = llm_config.get("provider", "anthropic").lower()
        self.model = llm_config.get("model", "claude-3-7-sonnet-20250219")
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.temperature = llm_config.get("temperature", 0.1)
        self.api_base = llm_config.get("api_base", None)
        
        # Initialize appropriate API client
        if self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package is required for Claude API access")
            self._init_anthropic_client()
        elif self.provider == "deepseek":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package is required for Deepseek API access")
            if not deepseek_config:
                raise ValueError("Deepseek configuration is required when using Deepseek provider")
            self._init_deepseek_client()
        # Update the provider initialization to include local deepseek
        elif self.provider == "deepseek_local":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package is required for local Deepseek API access")
            if not deepseek_local_config:
                raise ValueError("Local Deepseek configuration is required when using local Deepseek provider")
            self._init_local_deepseek()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        # Analysis settings
        self.chunk_size = analysis_config.get("chunk_size", 50)
        self.overlap = analysis_config.get("overlap", 10)
        self.timeout = analysis_config.get("timeout", 300)
        self.max_retries = analysis_config.get("max_retries", 3)
        
        # Performance metrics
        self.api_calls = 0
        self.total_tokens = 0
        self.total_duration = 0
    
    def _init_anthropic_client(self):
        """Initialize Anthropic API client for Claude."""
        self.anthropic = Anthropic(api_key=self.llm_config["api_key"])
        if self.api_base:
            # Set custom API base URL if provided
            self.anthropic.base_url = self.api_base
        self.logger.info(f"Initialized Anthropic client for model: {self.model}")
    
    def _init_deepseek_client(self):
        """Initialize OpenAI-compatible client for Deepseek."""
        # Use Deepseek config if available, otherwise fall back to llm_config
        config = self.deepseek_config or self.llm_config
        
        # Get the API key and base URL
        api_key = config.get("api_key")
        api_base = config.get("api_base", "https://api.deepseek.com/v1")
        
        # Validate required config
        if not api_key:
            raise ValueError("Missing API key for Deepseek. Please provide it in the configuration.")
        
        # Override model with Deepseek model if specified
        if config.get("model"):
            self.model = config.get("model")
        
        self.logger.info(f"Initialized Deepseek client for model: {self.model}")
        self.logger.info(f"Using Deepseek API at: {api_base}")

    def _init_local_deepseek(self):
        """Initialize OpenAI-compatible client for local Deepseek inference."""
        # Use local deepseek config if available
        config = self.deepseek_local_config or self.llm_config
        
        # Set the model and base URL for local inference server
        self.model = config.get("model", "/home/akulg2/edr-llm-analyzer/models/mistral-7b-instruct")
        self.api_base = config.get("api_base", "http://localhost:8000/v1")
        
        # Enable performance measurement if configured
        self.measure_performance = config.get("measure_performance", False)
        
        self.logger.info(f"Initialized local Deepseek client for model: {self.model}")
        self.logger.info(f"Using inference server at: {self.api_base}")
        
        # Check available endpoints by first testing the base URL
        try:
            import requests
            from urllib.parse import urlparse
            
            # Parse the URL to get the base
            parsed_url = urlparse(self.api_base)
            base_server_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # First check if server is running at all
            self.logger.info(f"Testing connection to server at {base_server_url}")
            server_response = requests.get(base_server_url, timeout=5)
            
            if server_response.status_code >= 400:
                self.logger.warning(f"Server returned error code {server_response.status_code}")
            else:
                self.logger.info(f"Server responded with status code {server_response.status_code}")
                
            # Check common API endpoints to find the right one
            endpoints_to_try = [
                "/v1/chat/completions",  # OpenAI standard
                "/v1/completions",       # Older OpenAI style
                "/generate",             # Some vLLM servers
                "/api/v1/generate",      # Alternative format
                ""                       # Root endpoint
            ]
            
            found_endpoints = []
            for endpoint in endpoints_to_try:
                try:
                    test_url = f"{base_server_url}{endpoint}"
                    # Use HEAD request to check if endpoint exists without generating content
                    self.logger.debug(f"Testing endpoint: {test_url}")
                    response = requests.head(test_url, timeout=2)
                    
                    # Some servers return 405 Method Not Allowed for HEAD on valid endpoints
                    if response.status_code < 404 or response.status_code == 405:
                        found_endpoints.append(endpoint)
                        self.logger.info(f"Found potential API endpoint: {endpoint}")
                except Exception as e:
                    self.logger.debug(f"Failed testing endpoint {endpoint}: {e}")
            
            # If we found endpoints, update the API base URL
            if found_endpoints:
                # Prefer /v1/chat/completions if available
                if "/v1/chat/completions" in found_endpoints:
                    self.api_path = "/v1/chat/completions"
                else:
                    # Otherwise use the first found endpoint
                    self.api_path = found_endpoints[0]
                    
                self.logger.info(f"Will use API endpoint: {self.api_path}")
            else:
                self.logger.warning("No valid API endpoints found, will use default /v1/chat/completions")
                self.api_path = "/v1/chat/completions"
                
        except Exception as e:
            self.logger.warning(f"Error during endpoint discovery: {e}")
            self.logger.warning("Will use default API endpoint: /v1/chat/completions")
            self.api_path = "/v1/chat/completions"
            
        # Set a flag to indicate we're initialized
        self.is_initialized = True

    
    def _measure_resource_usage(self, tag: str = ""):
        """
        Measure current system resource usage.
        
        Args:
            tag: Optional tag to identify the measurement point
        """
        if not self.measure_performance or not RESOURCE_MONITORING_AVAILABLE:
            return
        
        # CPU measurements - more detailed
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_current = cpu_freq.current if cpu_freq else None
        
        # Per-core CPU usage
        per_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Memory measurements
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # GPU measurements if available
        gpu_info = {}
        if GPU_MONITORING_AVAILABLE and torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info[f"gpu_{i}"] = {
                    "name": gpu.name,
                    "memory_used_percent": gpu.memoryUtil * 100,
                    "utilization_percent": gpu.load * 100,
                    "temperature": gpu.temperature,  # Added temperature monitoring
                    "memory_used_MB": gpu.memoryUsed,
                    "memory_total_MB": gpu.memoryTotal
                }
        elif torch.cuda.is_available():
            # Fallback if GPUtil not available but CUDA is
            for i in range(torch.cuda.device_count()):
                gpu_info[f"gpu_{i}"] = {
                    "memory_allocated_MB": torch.cuda.memory_allocated(i) / (1024 * 1024),
                    "memory_reserved_MB": torch.cuda.memory_reserved(i) / (1024 * 1024),
                    "name": torch.cuda.get_device_name(i)
                }
        
        # I/O measurements
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        
        # Record the measurement
        measurement = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "cpu_percent": cpu_percent,
            "cpu_per_core": per_cpu_percent,
            "cpu_freq_mhz": cpu_freq_current,
            "memory_percent": memory.percent,
            "memory_used_GB": memory.used / (1024 * 1024 * 1024),
            "swap_percent": swap.percent,
            "disk_percent": disk_usage.percent,
            "gpu_info": gpu_info
        }
        
        self.resource_usage.append(measurement)
        self.logger.debug(f"Resource usage at {tag}: CPU: {cpu_percent}%, RAM: {memory.percent}%, "
                        f"GPU mem: {gpu_info.get('gpu_0', {}).get('memory_used_percent', 'N/A')}%")
        
    def _summarize_resource_usage(self) -> Dict[str, Any]:
        """
        Summarize collected resource usage metrics.
        
        Returns:
            Dictionary with resource usage summary
        """
        if not self.resource_usage:
            return {}
        
        # Calculate CPU usage statistics
        cpu_percentages = [m["cpu_percent"] for m in self.resource_usage]
        
        # Calculate memory usage statistics
        memory_percentages = [m["memory_percent"] for m in self.resource_usage]
        memory_used_GB = [m["memory_used_GB"] for m in self.resource_usage]
        
        # GPU statistics if available
        gpu_summaries = {}
        
        # Check if we have any GPU measurements
        has_gpu_info = any(bool(m.get("gpu_info", {})) for m in self.resource_usage)
        
        if has_gpu_info:
            # Get all unique GPU IDs across all measurements
            gpu_ids = set()
            for measurement in self.resource_usage:
                gpu_info = measurement.get("gpu_info", {})
                gpu_ids.update(gpu_info.keys())
            
            # Calculate statistics for each GPU
            for gpu_id in gpu_ids:
                # Get measurements for this GPU
                gpu_measurements = []
                for measurement in self.resource_usage:
                    if gpu_id in measurement.get("gpu_info", {}):
                        gpu_measurements.append(measurement["gpu_info"][gpu_id])
                
                if not gpu_measurements:
                    continue
                
                # Check which metrics are available in the measurements
                sample_measurement = gpu_measurements[0]
                
                if "memory_used_percent" in sample_measurement:
                    # We have percentage-based measurements
                    memory_util = [m.get("memory_used_percent", 0) for m in gpu_measurements]
                    utilization = [m.get("utilization_percent", 0) for m in gpu_measurements]
                    
                    gpu_summaries[gpu_id] = {
                        "name": sample_measurement.get("name", gpu_id),
                        "memory_utilization_percent": {
                            "min": min(memory_util),
                            "max": max(memory_util),
                            "mean": sum(memory_util) / len(memory_util)
                        },
                        "compute_utilization_percent": {
                            "min": min(utilization),
                            "max": max(utilization),
                            "mean": sum(utilization) / len(utilization)
                        }
                    }
                elif "memory_allocated_MB" in sample_measurement:
                    # We have absolute measurements (torch.cuda)
                    memory_allocated = [m.get("memory_allocated_MB", 0) for m in gpu_measurements]
                    memory_reserved = [m.get("memory_reserved_MB", 0) for m in gpu_measurements]
                    
                    gpu_summaries[gpu_id] = {
                        "memory_allocated_MB": {
                            "min": min(memory_allocated),
                            "max": max(memory_allocated),
                            "mean": sum(memory_allocated) / len(memory_allocated)
                        },
                        "memory_reserved_MB": {
                            "min": min(memory_reserved),
                            "max": max(memory_reserved),
                            "mean": sum(memory_reserved) / len(memory_reserved)
                        }
                    }
        
        # Find peak memory usage points
        peak_memory_index = memory_percentages.index(max(memory_percentages))
        peak_memory_point = self.resource_usage[peak_memory_index]
        
        # Track processing times between measurements
        processing_times = {}
        for i in range(1, len(self.resource_usage)):
            prev = self.resource_usage[i-1]
            curr = self.resource_usage[i]
            
            if "tag" in prev and "tag" in curr and prev["tag"] and curr["tag"]:
                if prev["tag"].endswith("_start") and curr["tag"].endswith("_end"):
                    # Extract the common part (e.g., "chunk_0" from "chunk_0_start" and "chunk_0_end")
                    stage = prev["tag"].rsplit("_", 1)[0]
                    
                    if stage:
                        # Calculate time difference
                        try:
                            prev_time = datetime.fromisoformat(prev["timestamp"])
                            curr_time = datetime.fromisoformat(curr["timestamp"])
                            time_diff = (curr_time - prev_time).total_seconds()
                            
                            processing_times[stage] = time_diff
                        except (ValueError, TypeError):
                            # Skip if datetime parsing fails
                            pass
        
        # Return the summary
        return {
            "cpu_percent": {
                "min": min(cpu_percentages),
                "max": max(cpu_percentages),
                "mean": sum(cpu_percentages) / len(cpu_percentages)
            },
            "memory_percent": {
                "min": min(memory_percentages),
                "max": max(memory_percentages),
                "mean": sum(memory_percentages) / len(memory_percentages)
            },
            "memory_usage_GB": {
                "min": min(memory_used_GB),
                "max": max(memory_used_GB),
                "mean": sum(memory_used_GB) / len(memory_used_GB),
                "peak_timestamp": peak_memory_point.get("timestamp", ""),
                "peak_tag": peak_memory_point.get("tag", "")
            },
            "gpu_info": gpu_summaries,
            "processing_times_seconds": processing_times,
            "measurement_count": len(self.resource_usage),
            "first_measurement": self.resource_usage[0]["timestamp"] if self.resource_usage else "",
            "last_measurement": self.resource_usage[-1]["timestamp"] if self.resource_usage else ""
        }

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze EDR logs using the LLM.
        
        Args:
            data: DataFrame containing EDR events
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting LLM analysis of EDR data")
        
        if data.empty:
            self.logger.warning("Empty dataset. Skipping analysis.")
            return {"error": "Empty dataset"}
        
        start_time = time.time()
        
        # Start resource measurement if enabled
        self._measure_resource_usage("analysis_start")
        
        try:
            # Divide data into chunks for processing
            chunks = self._create_chunks(data)
            self.logger.info(f"Divided data into {len(chunks)} chunks for processing")
            
            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                # Add measurement calls at key points
                self._measure_resource_usage(f"chunk_{i}_start")

                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    chunk_result = self._process_chunk(chunk, i)
                    chunk_results.append(chunk_result)
                except Exception as chunk_error:
                    self.logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                    # Create a placeholder result to avoid breaking the entire analysis
                    chunk_results.append({
                        "chunk_index": i,
                        "error": str(chunk_error),
                        "structured_results": {
                            "parsing_error": f"Processing error: {str(chunk_error)}",
                            "identified_threats": [],
                            "suspicious_patterns": [],
                            "risk_assessment": {
                                "overall_risk_level": "Unknown",
                                "reasoning": f"Analysis failed due to error: {str(chunk_error)}"
                            },
                            "recommendations": ["Retry analysis with different parameters"],
                            "limitations": ["Analysis failed due to technical error"]
                        }
                    })
                
                self._measure_resource_usage(f"chunk_{i}_end")
            
            # If all chunks failed, report the error but continue with empty results
            if all("error" in result for result in chunk_results):
                self.logger.error("All chunks failed to process")
                
                # Calculate performance metrics for error reporting
                self.total_duration = time.time() - start_time
                
                return {
                    "error": "All chunks failed to process",
                    "identified_threats": [],
                    "suspicious_patterns": [],
                    "risk_assessment": {
                        "overall_risk_level": "Unknown",
                        "reasoning": "Analysis failed due to technical errors"
                    },
                    "recommendations": ["Try with different LLM provider or model"],
                    "limitations": ["Complete analysis failure due to technical errors"],
                    "performance": {
                        "api_calls": self.api_calls,
                        "total_tokens": self.total_tokens,
                        "duration_seconds": self.total_duration,
                        "error_rate": 1.0  # 100% errors
                    }
                }
            
            self._measure_resource_usage("combining_results_start")
            combined_results = self._combine_results(chunk_results)
            self._measure_resource_usage("combining_results_end")

            # Attempt to generate final analysis, but handle failures gracefully
            self._measure_resource_usage("final_analysis_start")
            try:
                final_analysis = self._generate_final_analysis(combined_results, data)
            except Exception as final_error:
                self.logger.error(f"Error generating final analysis: {final_error}")
                # Create basic final analysis with the combined results
                final_analysis = {
                    **combined_results,
                    "final_summary": {
                        "conclusion": "Analysis completed with errors",
                        "confidence": "Low",
                        "reasoning": f"Final analysis failed: {str(final_error)}"
                    }
                }
            self._measure_resource_usage("final_analysis_end")

            if self.measure_performance and self.resource_usage:
                final_analysis["resource_usage"] = self._summarize_resource_usage()
            
            # Calculate performance metrics
            self.total_duration = time.time() - start_time
            
            # Add analysis configuration to the results
            final_analysis["analysis_config"] = {
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "timeout": self.timeout,
                "max_retries": self.max_retries
            }
            
            # Add performance metrics to results
            final_analysis["performance"] = {
                "api_calls": self.api_calls,
                "total_tokens": self.total_tokens,
                "average_tokens_per_call": int(self.total_tokens / max(1, self.api_calls)),
                "duration_seconds": self.total_duration,
                "events_per_second": len(data) / self.total_duration,
                "cost_estimate": self._estimate_cost()
            }
            
            self.logger.info(f"Analysis completed with {self.api_calls} API calls in {self.total_duration:.2f} seconds")
            
            return final_analysis
            
        except Exception as e:
            self.logger.error(f"Error during LLM analysis: {e}", exc_info=True)
            
            # Calculate partial performance metrics for error reporting
            self.total_duration = time.time() - start_time
            
            return {
                "error": str(e),
                "identified_threats": [],
                "suspicious_patterns": [],
                "risk_assessment": {
                    "overall_risk_level": "Unknown",
                    "reasoning": f"Analysis failed: {str(e)}"
                },
                "recommendations": [],
                "limitations": ["Analysis failed due to technical error"],
                "performance": {
                    "api_calls": self.api_calls,
                    "total_tokens": self.total_tokens, 
                    "duration_seconds": self.total_duration,
                    "error": str(e)
                }
            }

        finally:
            # Final resource measurement
            self._measure_resource_usage("analysis_end")
            
            # Clean up resources
            if self.provider == "deepseek_local" and torch.cuda.is_available():
                # Force garbage collection to free up CUDA memory
                gc.collect()
                torch.cuda.empty_cache()
    
    def _create_chunks(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Divide the dataset into overlapping chunks for processing.
        
        Args:
            data: DataFrame containing EDR events
            
        Returns:
            List of DataFrames, each containing a chunk of events
        """
        chunks = []
        total_events = len(data)
        
        # Handle case where dataset is smaller than chunk size
        if total_events <= self.chunk_size:
            return [data]
        
        # Create overlapping chunks
        for i in range(0, total_events, self.chunk_size - self.overlap):
            end_idx = min(i + self.chunk_size, total_events)
            chunk = data.iloc[i:end_idx].copy()
            chunks.append(chunk)
            
            # Stop if we've reached the end
            if end_idx == total_events:
                break
        
        return chunks
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_index: int) -> Dict[str, Any]:
        """
        Process a single chunk of EDR events.
        
        Args:
            chunk: DataFrame containing a chunk of EDR events
            chunk_index: Index of the current chunk
            
        Returns:
            Dictionary containing analysis results for the chunk
        """
        try:
            # Convert chunk to format suitable for LLM
            self.logger.info(f"Processing chunk {chunk_index+1}, containing {len(chunk)} events")
            events_text = self._format_events_for_llm(chunk)
            
            # Safety check on events_text
            if events_text is None:
                self.logger.warning("Event formatting returned None for chunk {chunk_index+1}")
                events_text = f"No valid events were found in chunk {chunk_index+1}."
            
            # Craft prompt
            prompt = self._create_chunk_prompt(events_text, chunk_index)

            assert len(prompt) < 25000

            # Call LLM API
            response = self._call_llm_api(prompt)

            # Parse and structure the response
            structured_results = self._parse_llm_response(response, chunk)
            
            return {
                "chunk_index": chunk_index,
                "events": chunk.to_dict(orient="records") if not chunk.empty else [],
                "raw_response": response,
                "structured_results": structured_results,
                "event_range": {
                    "start": chunk.iloc[0]["timestamp"] if not chunk.empty and "timestamp" in chunk.columns else None,
                    "end": chunk.iloc[-1]["timestamp"] if not chunk.empty and "timestamp" in chunk.columns else None,
                    "start_index": chunk.index[0] if not chunk.empty else None,
                    "end_index": chunk.index[-1] if not chunk.empty else None
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_index+1}: {e}")
            # Rather than raising, return a structured error result
            return {
                "chunk_index": chunk_index,
                "error": str(e),
                "structured_results": {
                    "parsing_error": f"Processing error: {str(e)}",
                    "identified_threats": [],
                    "suspicious_patterns": [],
                    "risk_assessment": {
                        "overall_risk_level": "Unknown",
                        "reasoning": f"Analysis failed due to error: {str(e)}"
                    },
                    "recommendations": ["Retry analysis with different parameters or smaller chunk size"],
                    "limitations": ["Analysis failed due to technical error"]
                }
            }
        
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: Prompt string for the LLM
            
        Returns:
            Response string from the LLM
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Call appropriate API based on provider
                if self.provider == "anthropic":
                    response = self._call_anthropic_api(prompt)
                elif self.provider == "deepseek" or self.provider == "deepseek_local":
                    # Both Deepseek cloud and local use the same OpenAI-compatible API
                    response = self._call_deepseek_api(prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                # Update metrics
                elapsed = time.time() - start_time
                self.api_calls += 1
                
                self.logger.debug(f"LLM API call ({self.provider}) completed in {elapsed:.2f}s")
                
                return response
                
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt+1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt + np.random.random()
                    self.logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("Max retries exceeded")
                    raise
        
        # This should not be reached, but just in case
        raise RuntimeError("Max retries exceeded")
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """
        Call the Anthropic Claude API.
        
        Args:
            prompt: Prompt string for Claude
            
        Returns:
            Response string from Claude
        """
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=self.timeout
        )
        
        # Update token metrics
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        
        self.logger.debug(f"Claude API call completed with "
                         f"{response.usage.input_tokens} input tokens and "
                         f"{response.usage.output_tokens} output tokens")
        
        return response.content[0].text
    
    def _call_deepseek_api(self, prompt: str) -> str:
        """
        Call the Deepseek API using OpenAI-compatible interface.
        
        Args:
            prompt: Prompt string for Deepseek
            
        Returns:
            Response string from Deepseek
        """
        try:
            # Check if prompt is None or empty
            if prompt is None or prompt == "":
                self.logger.error("Empty prompt detected. Cannot send None as content to Deepseek API.")
                raise ValueError("Prompt cannot be None or empty")
                
            # Make sure we have the right base URL and API key set
            if self.provider == "deepseek":
                # Use Deepseek cloud configuration
                if self.deepseek_config:
                    api_base = self.deepseek_config.get("api_base", "https://api.deepseek.com/v1")
                    api_key = self.deepseek_config.get("api_key")
                else:
                    api_base = "https://api.deepseek.com/v1"
                    api_key = self.llm_config.get("api_key")
                
                # For cloud API, use standard endpoint
                api_path = "/chat/completions"
                
            else:  # deepseek_local
                # Use local inference server configuration
                api_base = self.api_base or "http://localhost:8000/v1"
                api_key = "dummy-key"  # Local servers often don't need real keys
                
                # Use the discovered endpoint path if available
                api_path = getattr(self, 'api_path', '/v1/chat/completions')
                
                # Strip /v1 from base URL if it's already there and api_path starts with /v1
                if api_base.endswith('/v1') and api_path.startswith('/v1'):
                    api_base = api_base[:-3]  # Remove trailing /v1
                    
            # Create full URL for the API call
            from urllib.parse import urlparse
            
            # Parse the API base to get components
            parsed_url = urlparse(api_base)
            
            # Create a new base URL with the scheme and netloc only
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Combine the base with the path parts, making sure not to duplicate /v1
            if api_path.startswith('/'):
                full_api_url = f"{base_url}{api_path}"
            else:
                full_api_url = f"{base_url}/{api_path}"
            
            self.logger.info(f"Using full API URL: {full_api_url}")
            
            # Set the client configuration for this specific call
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # Create message list with explicit safety checks
            messages = []
            
            # Only add user message if prompt is valid
            if isinstance(prompt, str) and prompt.strip():
                messages.append({"role": "user", "content": prompt})
            else:
                # Use a default prompt if the original is invalid
                self.logger.warning("Invalid prompt detected, using default prompt")
                messages.append({"role": "user", "content": "Analyze security events."})
            
            # Log the actual request being sent
            self.logger.debug(f"Sending request to {full_api_url} with model {self.model}")
            self.logger.debug(f"First 100 chars of user prompt: {messages[-1]['content'][:100]}...")
            
            # Try a direct HTTP request first as a fallback for non-standard APIs
            import requests
            import json

           
            # Try direct HTTP request as a fallback
            payload = {"model": "/home/akulg2/edr-llm-analyzer/models/mistral-7b-instruct","messages": messages, "max_tokens": 8192}
            

            headers = {"Content-Type": "application/json"}
            
            if api_key and api_key != "dummy-key":
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Make the HTTP request
            http_response = requests.post(full_api_url,headers=headers,json=payload,timeout=300)

            
            http_response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            
            # Parse the response JSON
            response_data = http_response.json()
            
            # Extract the generated text
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0] and "content" in response_data["choices"][0]["message"]:
                    output_text = response_data["choices"][0]["message"]["content"]
                elif "text" in response_data["choices"][0]:
                    output_text = response_data["choices"][0]["text"]
                else:
                    raise ValueError(f"Unexpected response format: {response_data}")
            else:
                raise ValueError(f"No choices found in response: {response_data}")
            
            # Estimate token usage
            estimated_input_tokens = len(prompt) / 4
            estimated_output_tokens = len(output_text) / 4
            self.total_tokens += estimated_input_tokens + estimated_output_tokens
            
            self.logger.debug(f"Direct HTTP request completed with estimated "
                            f"{estimated_input_tokens:.0f} input tokens and "
                            f"{estimated_output_tokens:.0f} output tokens")
            
            return output_text
        
        except Exception as e:
            self.logger.error(f"Error calling Deepseek API: {str(e)}")
            
            # Check for common errors and provide more helpful error messages
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                self.logger.error("Authentication error: Please check your API key")
            elif "not found" in str(e).lower() or "404" in str(e).lower():
                self.logger.error(f"API endpoint not found. The server may not support the OpenAI API format or may be using a different endpoint structure.")
                self.logger.error("Try checking the server documentation for the correct endpoint.")
            elif "timeout" in str(e).lower():
                self.logger.error(f"API request timed out after {self.timeout} seconds")
            elif "content" in str(e).lower() and "None" in str(e):
                self.logger.error("Invalid content: 'content' field cannot be None")
            
            # Provide a fallback response for testing
            if "TESTING_MODE" in os.environ:
                self.logger.warning("TESTING_MODE enabled, returning mock response")
                return self._get_mock_response()
            
            # Re-raise with more context
            raise RuntimeError(f"Deepseek API call failed: {str(e)}") from e

    def _get_mock_response(self):
        """Generate a mock response for testing purposes."""
        return """
        {
          "identified_threats": [
            {
              "threat_id": "mock-threat-1",
              "threat_type": "Mock Detection",
              "related_events": ["evt-1", "evt-2"],
              "severity": "Medium",
              "confidence": "Medium",
              "description": "This is a mock response because the LLM API call failed."
            }
          ],
          "suspicious_patterns": [],
          "risk_assessment": {
            "overall_risk_level": "Medium",
            "reasoning": "This is a mock response for testing."
          },
          "recommendations": [
            "Fix API connection issues",
            "Check server configuration"
          ],
          "limitations": [
            "API failure prevented proper analysis",
            "Results are mocked for testing continuity"
          ]
        }
        """
    
    def _parse_llm_response(self, response: str, chunk: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.
        
        Args:
            response: Response string from the LLM
            chunk: DataFrame containing the chunk of events
            
        Returns:
            Dictionary containing structured results
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in response")
                return {"parsing_error": "No JSON found in response", "raw_text": response}
            
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            # Add metadata
            result["_metadata"] = {
                "parsing_success": True,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {e}")
            return {
                "parsing_error": f"JSON parsing error: {str(e)}",
                "raw_text": response
            }
        except Exception as e:
            self.logger.warning(f"Error processing LLM response: {e}")
            return {
                "parsing_error": f"Processing error: {str(e)}",
                "raw_text": response
            }
    
    def _combine_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from multiple chunks.
        
        Args:
            chunk_results: List of dictionaries containing chunk results
            
        Returns:
            Dictionary containing combined results
        """
        self.logger.info(f"Combining results from {len(chunk_results)} chunks")
        
        all_threats = []
        all_patterns = []
        all_recommendations = set()
        all_limitations = set()
        
        # Track seen threat IDs to avoid duplicates
        seen_threat_ids = set()
        seen_pattern_ids = set()
        
        # Collect results from each chunk
        for i, result in enumerate(chunk_results):
            structured = result.get("structured_results", {})
            
            # Skip if parsing error
            if "parsing_error" in structured:
                self.logger.warning(f"Skipping chunk {i} due to parsing error")
                continue
            
            # Process threats
            for threat in structured.get("identified_threats", []):
                # Generate a unique ID if needed
                if "threat_id" not in threat or not threat["threat_id"]:
                    threat["threat_id"] = f"threat-{len(seen_threat_ids)+1}"
                
                # Skip if duplicate
                if threat["threat_id"] in seen_threat_ids:
                    continue
                
                # Add chunk info
                threat["source_chunk"] = i
                seen_threat_ids.add(threat["threat_id"])
                all_threats.append(threat)
            
            # Process patterns
            for pattern in structured.get("suspicious_patterns", []):
                # Generate a unique ID if needed
                if "pattern_id" not in pattern or not pattern["pattern_id"]:
                    pattern["pattern_id"] = f"pattern-{len(seen_pattern_ids)+1}"
                
                # Skip if duplicate
                if pattern["pattern_id"] in seen_pattern_ids:
                    continue
                
                # Add chunk info
                pattern["source_chunk"] = i
                seen_pattern_ids.add(pattern["pattern_id"])
                all_patterns.append(pattern)
            
            # Collect recommendations and limitations
            for rec in structured.get("recommendations", []):
                all_recommendations.add(rec)
                
            for lim in structured.get("limitations", []):
                all_limitations.add(lim)
        
        # Assess overall risk
        overall_risk = self._assess_overall_risk(all_threats)
        
        return {
            "identified_threats": all_threats,
            "suspicious_patterns": all_patterns,
            "risk_assessment": overall_risk,
            "recommendations": list(all_recommendations),
            "limitations": list(all_limitations),
            "chunk_count": len(chunk_results),
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_overall_risk(self, threats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the overall risk based on identified threats.
        
        Args:
            threats: List of threat dictionaries
            
        Returns:
            Dictionary containing risk assessment
        """
        # Map severity strings to numeric values
        severity_map = {"High": 3, "Medium": 2, "Low": 1}
        
        # Count threats by severity
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for threat in threats:
            severity = threat.get("severity", "Low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Simple risk calculation
        if severity_counts["High"] > 0:
            risk_level = "High"
        elif severity_counts["Medium"] > 0:
            risk_level = "Medium"
        elif severity_counts["Low"] > 0:
            risk_level = "Low"
        else:
            risk_level = "Minimal"
        
        reasoning = (
            f"Based on {len(threats)} identified threats "
            f"({severity_counts['High']} high, {severity_counts['Medium']} medium, "
            f"{severity_counts['Low']} low severity)."
        )
        
        return {
            "overall_risk_level": risk_level,
            "reasoning": reasoning,
            "threat_count": len(threats),
            "severity_distribution": severity_counts
        }
    
    def _generate_final_analysis(self, combined_results: Dict[str, Any], 
                               original_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a final comprehensive analysis using the combined results.
        
        Args:
            combined_results: Dictionary containing combined results from all chunks
            original_data: Original DataFrame containing all EDR events
            
        Returns:
            Dictionary containing final analysis
        """
        self.logger.info("Generating final analysis")
        
        # Extract key information
        threats = combined_results["identified_threats"]
        patterns = combined_results["suspicious_patterns"]
        
        # Skip if no threats or patterns found
        if not threats and not patterns:
            self.logger.info("No threats or suspicious patterns detected")
            return {
                **combined_results,
                "attack_timeline": [],
                "potential_attack_chains": [],
                "final_summary": {
                    "conclusion": "No significant threats detected",
                    "confidence": "Medium",
                    "reasoning": "Analysis did not reveal any significant security threats in the provided EDR logs."
                }
            }
        
        # Identify potential attack chains by connecting related threats
        attack_chains = self._identify_attack_chains(threats, patterns)
        
        # Create a timeline of significant events
        timeline = self._create_event_timeline(threats, patterns, original_data)
        
        # Generate summary prompt for final analysis
        summary_prompt = self._create_summary_prompt(combined_results, attack_chains, timeline)
        
        # Call LLM API for final analysis
        summary_response = self._call_llm_api(summary_prompt)
        
        try:
            # Parse the summary response
            json_start = summary_response.find('{')
            json_end = summary_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in summary response")
                summary = {
                    "conclusion": "Analysis complete but summary parsing failed",
                    "confidence": "Low",
                    "reasoning": "Unable to parse the final summary response from the LLM."
                }
            else:
                summary = json.loads(summary_response[json_start:json_end])
        except Exception as e:
            self.logger.warning(f"Error parsing summary response: {e}")
            summary = {
                "conclusion": "Analysis complete but summary parsing failed",
                "confidence": "Low",
                "reasoning": f"Error parsing summary: {str(e)}"
            }
        
        # Compile final results
        final_results = {
            **combined_results,
            "attack_timeline": timeline,
            "potential_attack_chains": attack_chains,
            "final_summary": summary,
            "raw_summary_response": summary_response
        }
        
        return final_results
    
    def _identify_attack_chains(self, threats: List[Dict[str, Any]], 
                             patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify potential attack chains by connecting related threats and patterns.
        
        Args:
            threats: List of threat dictionaries
            patterns: List of pattern dictionaries
            
        Returns:
            List of dictionaries representing potential attack chains
        """
        attack_chains = []
        processed_threats = set()
        
        # Group threats by related events
        event_to_threats = {}
        for i, threat in enumerate(threats):
            for event_id in threat.get("related_events", []):
                if event_id not in event_to_threats:
                    event_to_threats[event_id] = []
                event_to_threats[event_id].append(i)
        
        # Build chains based on shared events
        for i, threat in enumerate(threats):
            # Skip if already processed
            if i in processed_threats:
                continue
            
            # Start a new chain
            chain = {
                "chain_id": f"chain-{len(attack_chains)+1}",
                "threats": [threat],
                "patterns": [],
                "related_events": set(threat.get("related_events", [])),
                "severity": threat.get("severity", "Low")
            }
            
            # Mark as processed
            processed_threats.add(i)
            
            # Find connected threats
            connected_threats = set()
            for event_id in chain["related_events"]:
                connected_threats.update(event_to_threats.get(event_id, []))
            
            # Add connected threats to chain
            for j in connected_threats:
                if j != i and j not in processed_threats:
                    chain["threats"].append(threats[j])
                    chain["related_events"].update(threats[j].get("related_events", []))
                    
                    # Update chain severity to highest severity
                    threat_severity = threats[j].get("severity", "Low")
                    if self._severity_to_number(threat_severity) > self._severity_to_number(chain["severity"]):
                        chain["severity"] = threat_severity
                    
                    processed_threats.add(j)
            
            # Add related patterns
            for pattern in patterns:
                pattern_events = set(pattern.get("related_events", []))
                if pattern_events.intersection(chain["related_events"]):
                    chain["patterns"].append(pattern)
            
            # Finalize chain
            chain["related_events"] = list(chain["related_events"])
            chain["event_count"] = len(chain["related_events"])
            chain["threat_count"] = len(chain["threats"])
            chain["pattern_count"] = len(chain["patterns"])
            
            attack_chains.append(chain)
        
        # Sort chains by severity
        attack_chains.sort(key=lambda c: self._severity_to_number(c["severity"]), reverse=True)
        
        return attack_chains
    
    def _severity_to_number(self, severity: str) -> int:
        """Convert severity string to numeric value for comparison."""
        severity_map = {"High": 3, "Medium": 2, "Low": 1}
        return severity_map.get(severity, 0)
    
    def _create_event_timeline(self, threats: List[Dict[str, Any]], 
                             patterns: List[Dict[str, Any]],
                             data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create a timeline of significant events based on threats and patterns.
        
        Args:
            threats: List of threat dictionaries
            patterns: List of pattern dictionaries
            data: DataFrame containing all EDR events
            
        Returns:
            List of dictionaries representing the timeline
        """
        # Collect all relevant event IDs
        relevant_events = set()
        for threat in threats:
            relevant_events.update(threat.get("related_events", []))
        
        for pattern in patterns:
            relevant_events.update(pattern.get("related_events", []))
        
        # Convert to list
        relevant_events = list(relevant_events)
        
        # Filter data to only include relevant events
        timeline_data = data[data["event_id"].isin(relevant_events)].copy()
        
        # Sort by timestamp
        if "timestamp" in timeline_data.columns:
            timeline_data = timeline_data.sort_values("timestamp")
        
        # Build timeline
        timeline = []
        for _, row in timeline_data.iterrows():
            event = row.to_dict()
            
            # Find related threats and patterns
            related_threats = []
            for threat in threats:
                if event["event_id"] in threat.get("related_events", []):
                    related_threats.append(threat["threat_id"])
            
            related_patterns = []
            for pattern in patterns:
                if event["event_id"] in pattern.get("related_events", []):
                    related_patterns.append(pattern["pattern_id"])
            
            # Add to timeline
            timeline.append({
                "event_id": event["event_id"],
                "timestamp": event.get("timestamp", ""),
                "event_type": event.get("event_type", "unknown"),
                "description": f"{event.get('process_name', '')} {event.get('command_line', '')}".strip(),
                "hostname": event.get("hostname", ""),
                "username": event.get("username", ""),
                "severity": event.get("severity", 0),
                "related_threats": related_threats,
                "related_patterns": related_patterns
            })
        
        return timeline
    
    def _create_summary_prompt(self, combined_results: Dict[str, Any],
                             attack_chains: List[Dict[str, Any]],
                             timeline: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for the final summary analysis.
        
        Args:
            combined_results: Dictionary containing combined results from all chunks
            attack_chains: List of dictionaries representing potential attack chains
            timeline: List of dictionaries representing the event timeline
            
        Returns:
            Prompt string for the LLM
        """
        # Extract key information
        threats = combined_results["identified_threats"]
        patterns = combined_results["suspicious_patterns"]
        recommendations = combined_results["recommendations"]
        limitations = combined_results["limitations"]
        
        # Format timeline for readability
        timeline_str = "Key events timeline:\n"
        for i, event in enumerate(timeline[:10]):  # Limit to 10 events for brevity
            timestamp = event.get("timestamp", "unknown time")
            if isinstance(timestamp, (pd.Timestamp, datetime)):
                timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
            timeline_str += (
                f"  Event {i+1}: {timestamp} - {event.get('event_type', '')} - "
                f"{event.get('description', '')}\n"
            )
        
        if len(timeline) > 10:
            timeline_str += f"  ... and {len(timeline) - 10} more events\n"
        
        # Format attack chains for readability
        chains_str = "Potential attack chains:\n"
        for i, chain in enumerate(attack_chains):
            chains_str += (
                f"  Chain {i+1} (Severity: {chain['severity']}): "
                f"{chain['threat_count']} threats across {chain['event_count']} events\n"
            )
            for j, threat in enumerate(chain["threats"]):
                chains_str += f"    - Threat: {threat.get('threat_type', 'Unknown')} - {threat.get('description', '')[:100]}...\n"
        
        prompt = f"""<instructions>
        You are a security analyst finalizing an EDR log analysis report. Based on the detailed analysis results below, provide a concise executive summary of the security situation.

        ANALYSIS RESULTS:
        - Identified {len(threats)} potential threats and {len(patterns)} suspicious patterns
        - Risk Assessment: {combined_results['risk_assessment'].get('overall_risk_level', 'Unknown')}
        - Reasoning: {combined_results['risk_assessment'].get('reasoning', '')}

        {timeline_str}

        {chains_str}

        Limitations of the analysis:
        {chr(10).join(f"- {lim}" for lim in limitations)}

        Please provide your final analysis in the following JSON format:
        ```jsons
        {{
          "conclusion": "string - A concise assessment of the security situation (1-2 sentences)",
          "confidence": "string - High, Medium, or Low",
          "attack_summary": "string - Brief description of the attack (if any) including tactics and impact (2-3 sentences)",
          "key_indicators": ["string - List of the most significant indicators of compromise"],
          "false_positives": ["string - Potential false positives in the analysis"],
          "reasoning": "string - Your reasoning for this conclusion (2-3 sentences)",
          "limitations": ["string - Key limitations of this LLM-based analysis approach"]
        }}
        ```

        Focus particularly on the limitations of using an LLM for this type of security analysis, including issues with context window constraints, probabilistic correlation, and resource requirements.
        </instructions>

        Based on the provided analysis results, generate a concise executive summary in the requested JSON format."""
        
        return prompt
    
    def _estimate_cost(self) -> Dict[str, float]:
        """
        Estimate the cost of the LLM API calls.
        
        Returns:
            Dictionary containing cost estimates
        """
        # Define pricing for different providers and models
        pricing = {
            "anthropic": {
                "claude-3-7-sonnet-20250219": {
                    "input_price_per_1k": 0.015,  # $0.015 per 1K input tokens
                    "output_price_per_1k": 0.075,  # $0.075 per 1K output tokens
                },
                "claude-3-opus": {
                    "input_price_per_1k": 0.03,  # $0.03 per 1K input tokens
                    "output_price_per_1k": 0.15,  # $0.15 per 1K output tokens
                },
                "claude-3-5-sonnet": {
                    "input_price_per_1k": 0.01,  # $0.01 per 1K input tokens
                    "output_price_per_1k": 0.05,  # $0.05 per 1K output tokens
                },
                "default": {
                    "input_price_per_1k": 0.015,  # Default pricing
                    "output_price_per_1k": 0.075,
                }
            },
            "deepseek": {
                "deepseek-chat": {
                    "input_price_per_1k": 0.0020,  # $0.0020 per 1K input tokens
                    "output_price_per_1k": 0.0060,  # $0.0060 per 1K output tokens
                },
                "deepseek-coder": {
                    "input_price_per_1k": 0.0025,  # $0.0025 per 1K input tokens
                    "output_price_per_1k": 0.0075,  # $0.0075 per 1K output tokens
                },
                "default": {
                    "input_price_per_1k": 0.0020,  # Default pricing
                    "output_price_per_1k": 0.0060,
                }
            },

            "deepseek_local": {
                "default": {
                    "input_price_per_1k": 0.0,  # Local models don't have API costs
                    "output_price_per_1k": 0.0,
                }
            },
            "default": {
                "default": {
                    "input_price_per_1k": 0.01,
                    "output_price_per_1k": 0.05,
                }
            }
        }
        
        # Get pricing for the current provider and model
        provider_pricing = pricing.get(self.provider, pricing["default"])
        model_pricing = provider_pricing.get(self.model, provider_pricing.get("default"))
        
        input_price_per_1k = model_pricing["input_price_per_1k"]
        output_price_per_1k = model_pricing["output_price_per_1k"]
        
        # Estimate input and output tokens
        # For Claude, we have exact counts. For others, we estimate
        if self.provider == "anthropic":
            # For Claude, we have the exact token counts
            input_tokens = self.total_tokens * 0.7  # Approximation: 70% input
            output_tokens = self.total_tokens * 0.3  # Approximation: 30% output
        else:
            # For other providers, use the same approximation if we don't have exact counts
            input_tokens = self.total_tokens * 0.7
            output_tokens = self.total_tokens * 0.3
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }

    
    def _format_events_for_llm(self, events: pd.DataFrame) -> str:
        """
        Format network flow events for LLM consumption.
        
        Args:
            events: DataFrame containing network flow events
            
        Returns:
            Formatted string representation of events
        """
        try:
            if events is None or events.empty:
                return "No events available for analysis."

            formatted_result = f"NETWORK FLOW DATA ({len(events)} total events)\n\n"
            
            # Process each event
            for i, (_, row) in enumerate(events.iterrows()):
                event_str = (f"Event {i+1} (ID: {row.get('event_id', f'unknown-{i}')})\n"
                            f"Time: {row.get('timestamp', 'unknown')}\n"
                            f"Source: {row.get('source', 'unknown')}\n"
                            f"Destination: {row.get('destination', 'unknown')}\n"
                            f"Action: {row.get('action', 'unknown')}\n\n")
                
                formatted_result += event_str
                
            return formatted_result

        except Exception as e:
            self.logger.error(f"Error formatting events for LLM: {e}")
            # Return a default message
            return f"Error formatting events: {str(e)}. No events could be processed."
    
        

    def _create_chunk_prompt(self, events_text: str, chunk_index: int) -> str:
        """
        Create a prompt for analyzing network flow data.
        
        Args:
            events_text: Formatted string representation of network flow events
            chunk_index: Index of the current chunk
            
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""You are a security analyst reviewing network flow data (Chunk #{chunk_index+1}):

        {events_text}

        Each event represents an information flow in the system with the following format:
        - Source: The process or program initiating the action
        - Destination: The target of the action (typically IP:port)
        - Action: The type of operation performed
        
        Analyze these events for security threats and attack patterns. Look for:
        - Suspicious process behavior (unusual processes making connections)
        - Unusual network connections (connections to suspicious IPs/ports)
        - Pattern of lateral movement (internal network scanning or connections)
        - Data exfiltration patterns (large outbound flows, unusual destinations)
        - Command and control (C2) communication patterns (regular beaconing, unusual ports)
        - Persistence mechanisms
        
        Identify chains of related events that might represent stages of an attack. Pay special attention to:
        - Processes connecting to known malicious IPs
        - Multiple failed connection attempts
        - Unusual protocols or ports
        - Sequential patterns that might indicate multi-stage attacks
        - Same source connecting to multiple destinations in rapid succession (scanning)
        - Same destination receiving connections from multiple internal sources (lateral movement)

        Return your analysis in the following JSON format:
        ```json
        {{
          "identified_threats": [
            {{
              "threat_id": "THREAT-{chunk_index+1}-1",
              "threat_type": "Type of threat (e.g., C2 Communication, Data Exfiltration, Lateral Movement)",
              "related_events": ["event_id1", "event_id2"],
              "severity": "High/Medium/Low",
              "confidence": "High/Medium/Low",
              "description": "Detailed description of the threat"
            }}
          ],
          "suspicious_patterns": [
            {{
              "pattern_id": "PATTERN-{chunk_index+1}-1",
              "pattern_type": "Type of pattern (e.g., Unusual Process Behavior, Network Scanning)",
              "related_events": ["event_id1", "event_id2"],
              "description": "Description of the suspicious pattern"
            }}
          ],
          "potential_attack_chains": [
            {{
              "chain_id": "CHAIN-{chunk_index+1}-1",
              "description": "Description of the attack chain sequence",
              "events": ["event_id1", "event_id2", "event_id3"],
              "severity": "High/Medium/Low",
              "techniques": ["Initial Access", "Execution", "Lateral Movement", "Data Exfiltration"]
            }}
          ],
          "risk_assessment": {{
            "overall_risk_level": "High/Medium/Low",
            "reasoning": "Brief reasoning for the risk assessment"
          }},
          "recommendations": ["Security recommendations"],
          "limitations": ["Analysis limitations"]
        }}
        ```"""

        return prompt
