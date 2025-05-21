#!/usr/bin/env python3
"""
Script to evaluate and benchmark the performance of LLM-based EDR log analysis.

This script calculates performance metrics and identifies limitations in the 
LLM's ability to analyze large security logs, specifically focused on:
1. Context window limitations
2. Probabilistic correlation accuracy
3. Resource requirements
4. Memory limitations in tracking attack patterns

Usage:
    python benchmark_results.py --results PATH_TO_RESULTS --output OUTPUT_PATH
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logging, format_duration

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM EDR analysis results")
    
    parser.add_argument(
        "--results", "-r", 
        required=True, 
        help="Path to the analysis results directory"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="./data/benchmark", 
        help="Path to save benchmark results (default: ./data/benchmark)"
    )
    
    parser.add_argument(
        "--ground-truth", "-g", 
        help="Path to ground truth file (if available)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def load_results(results_dir: Path) -> Dict[str, Any]:
    """
    Load analysis results from the specified directory.
    
    Args:
        results_dir: Directory containing analysis results
        
    Returns:
        Dictionary containing aggregated results and metrics
    """
    # Load the aggregated results file
    aggregated_path = results_dir / "aggregated_results.json"
    if aggregated_path.exists():
        with open(aggregated_path, 'r') as f:
            return json.load(f)
    
    # If no aggregated file, try to find final results
    result_files = list(results_dir.glob("analysis_results_*.json"))
    if result_files:
        # Use the most recent one
        result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        with open(result_files[0], 'r') as f:
            return json.load(f)
    
    # If no final results, try to combine individual chunk results
    chunk_files = list(results_dir.glob("chunk_result_*.json"))
    if chunk_files:
        # Sort by chunk number
        chunk_files.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        # Load all chunks
        chunks = []
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                chunks.append(json.load(f))
        
        # Combine chunks (simplified)
        all_threats = []
        for i, chunk in enumerate(chunks):
            for threat in chunk.get("identified_threats", []):
                threat["source_chunk"] = i
                all_threats.append(threat)
        
        # Create a simple combined result
        return {
            "identified_threats": all_threats,
            "chunk_count": len(chunks),
            "manually_combined": True
        }
    
    raise FileNotFoundError(f"No results found in {results_dir}")

def load_ground_truth(ground_truth_path: Path) -> Dict[str, Any]:
    """
    Load ground truth data for comparison if available.
    
    Args:
        ground_truth_path: Path to ground truth file
        
    Returns:
        Dictionary containing ground truth data
    """
    if not ground_truth_path:
        return None
    
    if not ground_truth_path.exists():
        logging.warning(f"Ground truth file not found: {ground_truth_path}")
        return None
    
    try:
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading ground truth data: {e}")
        return None

def analyze_context_window_limitations(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze limitations caused by LLM context window constraints.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary containing context window limitation metrics
    """
    logging.info("Analyzing context window limitations...")
    
    # Get chunk information
    chunk_count = results.get("chunk_count", 0)
    if chunk_count == 0:
        return {"error": "No chunk information available"}
    
    # Check for cross-chunk threats
    threats = results.get("identified_threats", [])
    threat_events = {}
    
    # Collect events related to each threat
    for threat in threats:
        threat_id = threat.get("threat_id", "unknown")
        related_events = set(threat.get("related_events", []))
        source_chunk = threat.get("source_chunk", "unknown")
        
        if threat_id not in threat_events:
            threat_events[threat_id] = {
                "events": related_events,
                "chunks": {source_chunk},
                "threat_type": threat.get("threat_type", "Unknown"),
                "severity": threat.get("severity", "Low")
            }
        else:
            threat_events[threat_id]["events"].update(related_events)
            threat_events[threat_id]["chunks"].add(source_chunk)
    
    # Count threats contained within a single chunk vs. spanning multiple chunks
    single_chunk_threats = sum(1 for t in threat_events.values() if len(t["chunks"]) == 1)
    multi_chunk_threats = sum(1 for t in threat_events.values() if len(t["chunks"]) > 1)
    
    # Calculate threat linkage across chunks
    total_possible_connections = 0
    successful_connections = 0
    
    # Analyze attack chains for cross-chunk connections
    chains = results.get("potential_attack_chains", [])
    
    if chains:
        chain_threat_counts = [len(chain.get("threats", [])) for chain in chains]
        avg_chain_length = sum(chain_threat_counts) / len(chains) if chains else 0
        max_chain_length = max(chain_threat_counts) if chains else 0
        
        # Check if chains span across chunks
        cross_chunk_chains = 0
        for chain in chains:
            chain_chunks = set()
            for threat in chain.get("threats", []):
                if isinstance(threat, dict) and "source_chunk" in threat:
                    chain_chunks.add(threat["source_chunk"])
            
            if len(chain_chunks) > 1:
                cross_chunk_chains += 1
    else:
        avg_chain_length = 0
        max_chain_length = 0
        cross_chunk_chains = 0
    
    return {
        "total_chunks": chunk_count,
        "total_threats": len(threats),
        "single_chunk_threats": single_chunk_threats,
        "multi_chunk_threats": multi_chunk_threats,
        "cross_chunk_connection_rate": multi_chunk_threats / len(threats) if threats else 0,
        "avg_chain_length": avg_chain_length,
        "max_chain_length": max_chain_length,
        "cross_chunk_chains": cross_chunk_chains,
        "limitation_score": 1 - (multi_chunk_threats / len(threats) if threats else 0)  # Higher means more limited
    }

def analyze_probabilistic_correlation(results: Dict[str, Any], 
                                     ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze limitations in probabilistic correlation of events.
    
    Args:
        results: Dictionary containing analysis results
        ground_truth: Optional ground truth data for comparison
        
    Returns:
        Dictionary containing probabilistic correlation metrics
    """
    logging.info("Analyzing probabilistic correlation limitations...")
    
    # Get threats and patterns
    threats = results.get("identified_threats", [])
    patterns = results.get("suspicious_patterns", [])
    
    # Analyze confidence levels in results
    confidence_levels = {}
    for threat in threats:
        confidence = threat.get("confidence", "Unknown")
        if confidence not in confidence_levels:
            confidence_levels[confidence] = 0
        confidence_levels[confidence] += 1
    
    # Calculate confidence distribution
    total_threats = len(threats)
    confidence_distribution = {
        level: count / total_threats if total_threats > 0 else 0
        for level, count in confidence_levels.items()
    }
    
    # Analyze consistency of threat identification across chunks
    threat_types_by_chunk = {}
    for threat in threats:
        source_chunk = threat.get("source_chunk", "unknown")
        threat_type = threat.get("threat_type", "Unknown")
        
        if source_chunk not in threat_types_by_chunk:
            threat_types_by_chunk[source_chunk] = set()
        
        threat_types_by_chunk[source_chunk].add(threat_type)
    
    # Calculate consistency score (Jaccard similarity between chunk threat types)
    chunk_similarity_scores = []
    chunk_ids = list(threat_types_by_chunk.keys())
    
    for i in range(len(chunk_ids)):
        for j in range(i+1, len(chunk_ids)):
            chunk1 = chunk_ids[i]
            chunk2 = chunk_ids[j]
            
            types1 = threat_types_by_chunk[chunk1]
            types2 = threat_types_by_chunk[chunk2]
            
            if not types1 or not types2:
                continue
            
            # Jaccard similarity: size of intersection / size of union
            similarity = len(types1.intersection(types2)) / len(types1.union(types2))
            chunk_similarity_scores.append(similarity)
    
    avg_similarity = sum(chunk_similarity_scores) / len(chunk_similarity_scores) if chunk_similarity_scores else 0
    
    # If ground truth is available, calculate detection accuracy
    accuracy_metrics = {}
    if ground_truth:
        true_threats = ground_truth.get("threats", [])
        true_threat_types = {t.get("type", "Unknown") for t in true_threats}
        detected_threat_types = {t.get("threat_type", "Unknown") for t in threats}
        
        # Calculate precision and recall for threat types
        true_positives = len(true_threat_types.intersection(detected_threat_types))
        false_positives = len(detected_threat_types - true_threat_types)
        false_negatives = len(true_threat_types - detected_threat_types)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy_metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    return {
        "confidence_distribution": confidence_distribution,
        "avg_chunk_similarity": avg_similarity,
        "inconsistency_rate": 1 - avg_similarity,  # Higher means more inconsistent
        "total_threats": total_threats,
        "unique_threat_types": len({t.get("threat_type", "Unknown") for t in threats}),
        "accuracy_metrics": accuracy_metrics
    }

def analyze_resource_requirements(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze resource requirements for LLM-based analysis.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary containing resource requirement metrics
    """
    logging.info("Analyzing resource requirements...")
    
    # Get performance metrics
    performance = results.get("performance", {})
    if not performance:
        return {"error": "No performance metrics available"}
    
    # Extract key metrics
    total_tokens = performance.get("total_tokens", 0)
    total_api_calls = performance.get("total_api_calls", 0) or performance.get("api_calls", 0)
    total_duration = performance.get("total_duration_seconds", 0) or performance.get("duration_seconds", 0)
    
    # Calculate derived metrics
    tokens_per_call = total_tokens / total_api_calls if total_api_calls > 0 else 0
    tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
    cost_estimate = performance.get("cost_estimate", {}).get("total_cost", 0)
    
    # Resource usage metrics if available
    resource_usage = results.get("resource_usage", {})
    
    # GPU memory usage
    gpu_info = resource_usage.get("gpu_info", {})
    max_gpu_memory = 0
    
    for gpu_id, stats in gpu_info.items():
        if "memory_allocated_MB" in stats:
            max_mem = stats["memory_allocated_MB"].get("max", 0)
            max_gpu_memory = max(max_gpu_memory, max_mem)
        elif "memory_utilization_percent" in stats:
            # Convert percentage to MB if we know the total memory
            max_percent = stats["memory_utilization_percent"].get("max", 0)
            max_gpu_memory = max(max_gpu_memory, max_percent)
    
    # CPU and memory usage
    cpu_percent = resource_usage.get("cpu_percent", {}).get("max", 0)
    memory_percent = resource_usage.get("memory_percent", {}).get("max", 0)
    memory_gb = resource_usage.get("memory_usage_GB", {}).get("max", 0)
    
    return {
        "total_tokens": total_tokens,
        "total_api_calls": total_api_calls,
        "total_duration_seconds": total_duration,
        "tokens_per_api_call": tokens_per_call,
        "tokens_per_second": tokens_per_second,
        "cost_estimate_usd": cost_estimate,
        "max_gpu_memory_mb": max_gpu_memory,
        "max_cpu_percent": cpu_percent,
        "max_memory_percent": memory_percent,
        "max_memory_gb": memory_gb,
        "human_duration": format_duration(total_duration),
        "scaling_estimate": {
            "tokens_per_gb": total_tokens / performance.get("input_file_size_gb", 1) if "input_file_size_gb" in performance else "unknown",
            "time_per_gb": total_duration / performance.get("input_file_size_gb", 1) if "input_file_size_gb" in performance else "unknown",
            "cost_per_gb": cost_estimate / performance.get("input_file_size_gb", 1) if "input_file_size_gb" in performance and cost_estimate else "unknown"
        }
    }

def analyze_memory_limitations(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze memory limitations in tracking complex attack patterns.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary containing memory limitation metrics
    """
    logging.info("Analyzing memory limitations...")
    
    # Get attack chains and threats
    chains = results.get("potential_attack_chains", [])
    threats = results.get("identified_threats", [])
    
    # No chains available
    if not chains:
        return {
            "error": "No attack chains identified",
            "memory_limitation_score": 1.0  # Maximum limitation
        }
    
    # Analyze chain complexity
    chain_lengths = [len(chain.get("threats", [])) for chain in chains]
    avg_chain_length = sum(chain_lengths) / len(chains) if chains else 0
    max_chain_length = max(chain_lengths) if chains else 0
    
    # Check for cross-chunk chains
    cross_chunk_chains = 0
    max_chunks_per_chain = 0
    
    for chain in chains:
        chain_chunks = set()
        for threat in chain.get("threats", []):
            if isinstance(threat, dict) and "source_chunk" in threat:
                chain_chunks.add(threat["source_chunk"])
            elif "source_chunk" in chain:
                chain_chunks.add(chain["source_chunk"])
        
        if len(chain_chunks) > 1:
            cross_chunk_chains += 1
        
        max_chunks_per_chain = max(max_chunks_per_chain, len(chain_chunks))
    # Calculate the ratio of threats that were successfully connected into chains
    threats_in_chains = set()
    for chain in chains:
        for threat in chain.get("threats", []):
            if isinstance(threat, dict) and "threat_id" in threat:
                threats_in_chains.add(threat["threat_id"])
            elif isinstance(threat, str):
                threats_in_chains.add(threat)
    
    # Calculate threat connectivity ratio
    threat_ids = {t.get("threat_id", f"unknown-{i}") for i, t in enumerate(threats)}
    chain_coverage = len(threats_in_chains) / len(threat_ids) if threat_ids else 0
    
    # Calculate memory limitation score
    # Higher score means more limited (worse ability to track complex patterns)
    chunk_count = results.get("chunk_count", 1)
    memory_limitation_score = 1 - (cross_chunk_chains / len(chains) if chains else 0)
    
    return {
        "total_chains": len(chains),
        "avg_chain_length": avg_chain_length,
        "max_chain_length": max_chain_length,
        "cross_chunk_chains": cross_chunk_chains,
        "cross_chunk_chains_percent": cross_chunk_chains / len(chains) if chains else 0,
        "max_chunks_per_chain": max_chunks_per_chain,
        "chain_threat_coverage": chain_coverage,
        "memory_limitation_score": memory_limitation_score
    }

def generate_report(benchmark_results: Dict[str, Any], output_dir: Path) -> Path:
    """
    Generate a comprehensive benchmark report.
    
    Args:
        benchmark_results: Dictionary containing benchmark metrics
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    logging.info("Generating benchmark report...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{timestamp}.md"
    
    # Start building the report
    report_lines = [
        "# LLM-based EDR Log Analysis Benchmark Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report evaluates the performance and limitations of using Large Language Models (LLMs) for analyzing Endpoint Detection and Response (EDR) logs. The benchmark focuses on four key areas of limitations:",
        "",
        "1. **Context Window Constraints:** The LLM's ability to analyze events across context window boundaries",
        "2. **Probabilistic Correlation:** The accuracy and consistency of threat correlation",
        "3. **Resource Requirements:** The computational and financial costs of LLM-based analysis",
        "4. **Memory Limitations:** The LLM's ability to track complex attack patterns across data chunks",
        "",
        "### Key Findings",
        ""
    ]
    
    # Add key findings summary
    context_score = benchmark_results.get("context_window", {}).get("limitation_score", 0)
    prob_score = benchmark_results.get("probabilistic_correlation", {}).get("inconsistency_rate", 0)
    memory_score = benchmark_results.get("memory_limitations", {}).get("memory_limitation_score", 0)
    
    # Calculate overall limitation score (average of the three main scores)
    overall_score = (context_score + prob_score + memory_score) / 3
    
    def score_to_text(score):
        if score < 0.25:
            return "Minimal limitation"
        elif score < 0.5:
            return "Moderate limitation"
        elif score < 0.75:
            return "Significant limitation"
        else:
            return "Severe limitation"
    
    report_lines.extend([
        f"- **Overall Limitation Score:** {overall_score:.2f} - {score_to_text(overall_score)}",
        f"- **Context Window Limitation:** {context_score:.2f} - {score_to_text(context_score)}",
        f"- **Probabilistic Correlation Limitation:** {prob_score:.2f} - {score_to_text(prob_score)}",
        f"- **Memory Limitation:** {memory_score:.2f} - {score_to_text(memory_score)}",
        "",
        f"- **Resource Usage:** {benchmark_results.get('resource_requirements', {}).get('human_duration', 'Unknown')} of processing time",
        f"- **Token Usage:** {benchmark_results.get('resource_requirements', {}).get('total_tokens', 0):,} tokens",
        ""
    ])
    
    # Add detailed sections
    # 1. Context Window Constraints
    report_lines.extend([
        "## 1. Context Window Constraints",
        "",
        "This section evaluates the LLM's ability to analyze events that span across multiple context windows.",
        ""
    ])
    
    context_results = benchmark_results.get("context_window", {})
    if "error" not in context_results:
        # Get total threats with a minimum value of 1 to avoid division by zero
        total_threats = max(context_results.get('total_threats', 0), 1)

        # Calculate percentages safely
        single_chunk_percent = 100 * context_results.get('single_chunk_threats', 0) / total_threats if total_threats > 0 else 0
        multi_chunk_percent = 100 * context_results.get('multi_chunk_threats', 0) / total_threats if total_threats > 0 else 0
        report_lines.extend([
            f"- **Total Chunks Processed:** {context_results.get('total_chunks', 0)}",
            f"- **Total Threats Identified:** {context_results.get('total_threats', 0)}",
            f"- **Single-Chunk Threats:** {context_results.get('single_chunk_threats', 0)} ({single_chunk_percent:.1f}%)",
            f"- **Multi-Chunk Threats:** {context_results.get('multi_chunk_threats', 0)} ({multi_chunk_percent:.1f}%)",
            f"- **Cross-Chunk Connection Rate:** {context_results.get('cross_chunk_connection_rate', 0):.2f}",
            f"- **Average Attack Chain Length:** {context_results.get('avg_chain_length', 0):.2f} threats",
            f"- **Maximum Attack Chain Length:** {context_results.get('max_chain_length', 0)} threats",
            f"- **Cross-Chunk Attack Chains:** {context_results.get('cross_chunk_chains', 0)}",
            "",
            "### Context Window Limitation Analysis",
            "",
            f"The LLM demonstrated a cross-chunk connection rate of {context_results.get('cross_chunk_connection_rate', 0):.2f}, indicating its ability to connect related events across context window boundaries. A score of 1.0 would mean perfect connectivity, while 0.0 would mean no connections across chunks.",
            "",
            f"**Limitation Score:** {context_results.get('limitation_score', 0):.2f} - {score_to_text(context_results.get('limitation_score', 0))}",
            "",
            "This suggests that the LLM's context window constraints are a " + score_to_text(context_results.get('limitation_score', 0)).lower() + " factor in its ability to analyze complex security events spanning across multiple chunks of data.",
            ""
        ])
    else:
        report_lines.extend([
            f"Error analyzing context window constraints: {context_results.get('error')}",
            ""
        ])
    
    
    # 2. Probabilistic Correlation
    report_lines.extend([
        "## 2. Probabilistic Correlation",
        "",
        "This section evaluates the accuracy and consistency of the LLM's threat correlation across data chunks.",
        ""
    ])
    
    prob_results = benchmark_results.get("probabilistic_correlation", {})
    if "error" not in prob_results:
        # Format confidence distribution
        conf_dist = prob_results.get("confidence_distribution", {})
        conf_lines = []
        for level, rate in conf_dist.items():
            conf_lines.append(f"  - {level}: {rate:.2f} ({int(rate * 100)}%)")
        
        report_lines.extend([
            f"- **Total Threats Identified:** {prob_results.get('total_threats', 0)}",
            f"- **Unique Threat Types:** {prob_results.get('unique_threat_types', 0)}",
            f"- **Average Chunk Similarity:** {prob_results.get('avg_chunk_similarity', 0):.2f}",
            "",
            "### Confidence Distribution",
            ""
        ] + conf_lines + [
            "",
            "### Probabilistic Correlation Analysis",
            "",
            f"The LLM showed an average chunk similarity of {prob_results.get('avg_chunk_similarity', 0):.2f}, indicating the consistency of threat identification across chunks. A value of 1.0 would mean perfect consistency, while 0.0 would mean completely different threats identified in each chunk.",
            "",
            f"**Inconsistency Rate:** {prob_results.get('inconsistency_rate', 0):.2f} - {score_to_text(prob_results.get('inconsistency_rate', 0))}",
            "",
            "This suggests that the probabilistic nature of LLM correlation is a " + score_to_text(prob_results.get('inconsistency_rate', 0)).lower() + " factor in its ability to consistently identify threats across the dataset.",
            ""
        ])
        
        # Add accuracy metrics if available
        accuracy = prob_results.get("accuracy_metrics", {})
        if accuracy:
            report_lines.extend([
                "### Accuracy Metrics (compared to ground truth)",
                "",
                f"- **Precision:** {accuracy.get('precision', 0):.2f}",
                f"- **Recall:** {accuracy.get('recall', 0):.2f}",
                f"- **F1 Score:** {accuracy.get('f1_score', 0):.2f}",
                ""
            ])
    else:
        report_lines.extend([
            f"Error analyzing probabilistic correlation: {prob_results.get('error')}",
            ""
        ])
    
    # 3. Resource Requirements
    report_lines.extend([
        "## 3. Resource Requirements",
        "",
        "This section evaluates the computational and financial costs of using LLMs for EDR log analysis.",
        ""
    ])
    
    resource_results = benchmark_results.get("resource_requirements", {})
    if "error" not in resource_results:
        report_lines.extend([
            f"- **Total Processing Time:** {resource_results.get('human_duration', 'Unknown')}",
            f"- **Total API Calls:** {resource_results.get('total_api_calls', 0):,}",
            f"- **Total Tokens Used:** {resource_results.get('total_tokens', 0):,}",
            f"- **Tokens per API Call:** {resource_results.get('tokens_per_api_call', 0):,.1f}",
            f"- **Processing Rate:** {resource_results.get('tokens_per_second', 0):,.1f} tokens/second",
            f"- **Estimated Cost:** ${resource_results.get('cost_estimate_usd', 0):,.2f} USD",
            "",
            "### Resource Utilization",
            "",
            f"- **Peak CPU Usage:** {resource_results.get('max_cpu_percent', 0):.1f}%",
            f"- **Peak Memory Usage:** {resource_results.get('max_memory_gb', 0):.2f} GB ({resource_results.get('max_memory_percent', 0):.1f}%)",
            f"- **Peak GPU Memory:** {resource_results.get('max_gpu_memory_mb', 0) / 1024:.2f} GB",
            "",
            "### Scaling Estimates",
            ""
        ])
        
        # Add scaling estimates
        scaling = resource_results.get("scaling_estimate", {})
        if scaling:
            tokens_per_gb = scaling.get("tokens_per_gb", "unknown")
            if tokens_per_gb != "unknown":
                tokens_per_gb = f"{tokens_per_gb:,.0f} tokens/GB"
            
            time_per_gb = scaling.get("time_per_gb", "unknown")
            if time_per_gb != "unknown":
                time_per_gb = format_duration(time_per_gb)
            
            cost_per_gb = scaling.get("cost_per_gb", "unknown")
            if cost_per_gb != "unknown":
                cost_per_gb = f"${cost_per_gb:,.2f}/GB"
            
            report_lines.extend([
                f"- **Tokens Required per GB of Log Data:** {tokens_per_gb}",
                f"- **Processing Time per GB of Log Data:** {time_per_gb}",
                f"- **Cost per GB of Log Data:** {cost_per_gb}",
                "",
                "### Resource Requirements Analysis",
                "",
                "The resource requirements for LLM-based analysis show significant computational and potential financial costs, especially when scaling to larger datasets.",
                "",
                f"For a typical enterprise environment generating 1TB of EDR logs monthly, the estimated processing cost would be approximately ${float(cost_per_gb.replace('$', '').replace('/GB', '')) * 1000 if cost_per_gb != 'unknown' else 'unknown'} per month, with processing time of {format_duration(float(time_per_gb) * 1000) if time_per_gb != 'unknown' else 'unknown'}.",
                ""
            ])
    else:
        report_lines.extend([
            f"Error analyzing resource requirements: {resource_results.get('error')}",
            ""
        ])
    
    # 4. Memory Limitations
    report_lines.extend([
        "## 4. Memory Limitations",
        "",
        "This section evaluates the LLM's ability to track complex attack patterns across data chunks.",
        ""
    ])
    
    memory_results = benchmark_results.get("memory_limitations", {})
    if "error" not in memory_results:
        report_lines.extend([
            f"- **Total Attack Chains:** {memory_results.get('total_chains', 0)}",
            f"- **Average Chain Length:** {memory_results.get('avg_chain_length', 0):.2f} threats",
            f"- **Maximum Chain Length:** {memory_results.get('max_chain_length', 0)} threats",
            f"- **Cross-Chunk Chains:** {memory_results.get('cross_chunk_chains', 0)} ({memory_results.get('cross_chunk_chains_percent', 0) * 100:.1f}%)",
            f"- **Maximum Chunks per Chain:** {memory_results.get('max_chunks_per_chain', 0)}",
            f"- **Chain Threat Coverage:** {memory_results.get('chain_threat_coverage', 0):.2f} ({memory_results.get('chain_threat_coverage', 0) * 100:.1f}%)",
            "",
            "### Memory Limitation Analysis",
            "",
            f"The LLM demonstrated a chain threat coverage of {memory_results.get('chain_threat_coverage', 0):.2f}, indicating its ability to connect related threats into coherent attack chains. A value of 1.0 would mean all threats were incorporated into chains, while 0.0 would mean no coherent chains were formed.",
            "",
            f"**Memory Limitation Score:** {memory_results.get('memory_limitation_score', 0):.2f} - {score_to_text(memory_results.get('memory_limitation_score', 0))}",
            "",
            "This suggests that the LLM's lack of memory across chunks is a " + score_to_text(memory_results.get('memory_limitation_score', 0)).lower() + " factor in its ability to track complex attack patterns that span across multiple data chunks.",
            ""
        ])
    else:
        report_lines.extend([
            f"Error analyzing memory limitations: {memory_results.get('error')}",
            ""
        ])
    
    # 5. Conclusion
    report_lines.extend([
        "## Conclusion",
        "",
        "This benchmark demonstrates several key limitations of using LLMs for EDR log analysis:",
        "",
        f"1. **Context Window Constraints** ({context_score:.2f}): The LLM's fixed context window limits its ability to analyze events that span across multiple chunks of data.",
        f"2. **Probabilistic Correlation** ({prob_score:.2f}): The probabilistic nature of LLM reasoning leads to inconsistencies in threat correlation and potential false positives/negatives.",
        f"3. **Resource Requirements**: LLM-based analysis requires significant computational resources and can be costly to scale.",
        f"4. **Memory Limitations** ({memory_score:.2f}): The LLM lacks persistent memory between chunks, making it difficult to track complex attack patterns that evolve over time.",
        "",
        f"**Overall Limitation Score: {overall_score:.2f} - {score_to_text(overall_score)}**",
        "",
        "### Recommendations",
        "",
        "Based on these findings, the following recommendations are made for using LLMs in security analysis:",
        "",
        "1. **Hybrid Approach**: Combine LLMs with traditional rule-based detection systems to leverage the strengths of both approaches.",
        "2. **Enhanced Context Management**: Develop techniques to more effectively manage context across chunks, such as maintaining a summary of previous findings.",
        "3. **Supervised Analysis**: Use LLMs as an augmentation tool for human analysts rather than a standalone solution.",
        "4. **Specialized Fine-tuning**: Train LLMs specifically for security analysis tasks to improve their performance in this domain.",
        "",
        "Despite their limitations, LLMs show promise as tools to assist security analysts in processing and understanding large volumes of EDR data, particularly in initial triage and pattern recognition tasks.",
        ""
    ])
    
    # Write the report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Benchmark report generated at {report_path}")
    return report_path

def generate_visualizations(benchmark_results: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate visualizations for the benchmark results.
    
    Args:
        benchmark_results: Dictionary containing benchmark metrics
        output_dir: Directory to save visualizations
    """
    logging.info("Generating benchmark visualizations...")
    
    # Create output directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Context Window Limitations Visualization
    context_results = benchmark_results.get("context_window", {})
    if "error" not in context_results:
        plt.figure(figsize=(10, 6))
        
        # Threat distribution by chunk boundary
        labels = ['Single-Chunk Threats', 'Multi-Chunk Threats']
        sizes = [
            context_results.get('single_chunk_threats', 0),
            context_results.get('multi_chunk_threats', 0)
        ]
        
        plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#3498db', '#e74c3c']
        )
        plt.axis('equal')
        plt.title('Threat Distribution by Chunk Boundary')
        
        # Save the figure
        plt.savefig(vis_dir / "context_window_limitations.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 2. Probabilistic Correlation Visualization
    prob_results = benchmark_results.get("probabilistic_correlation", {})
    if "error" not in prob_results:
        plt.figure(figsize=(10, 6))
        
        # Confidence distribution
        conf_dist = prob_results.get("confidence_distribution", {})
        
        if conf_dist:
            labels = list(conf_dist.keys())
            sizes = list(conf_dist.values())
            
            plt.bar(
                labels,
                sizes,
                color=['#2ecc71' if x == 'High' else '#f39c12' if x == 'Medium' else '#e74c3c' for x in labels]
            )
            
            plt.ylim(0, 1.0)
            plt.ylabel('Proportion of Threats')
            plt.title('LLM Confidence Distribution')
            
            # Add percentage labels
            for i, v in enumerate(sizes):
                plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
            
            # Save the figure
            plt.savefig(vis_dir / "probabilistic_correlation.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    # 3. Resource Requirements Visualization
    resource_results = benchmark_results.get("resource_requirements", {})
    if "error" not in resource_results:
        plt.figure(figsize=(12, 8))
        
        # Resource utilization
        resources = {
            'CPU Usage (%)': resource_results.get('max_cpu_percent', 0),
            'Memory Usage (%)': resource_results.get('max_memory_percent', 0),
            'GPU Memory (GB)': resource_results.get('max_gpu_memory_mb', 0) / 1024
        }
        
        if resources:
            # Create a horizontal bar chart
            plt.barh(
                list(resources.keys()),
                list(resources.values()),
                color=['#3498db', '#9b59b6', '#f1c40f']
            )
            
            plt.xlabel('Resource Utilization')
            plt.title('Peak Resource Utilization')
            
            # Add value labels
            for i, v in enumerate(resources.values()):
                if i == 2:  # GPU Memory in GB
                    plt.text(v + 0.1, i, f"{v:.2f} GB", va='center')
                else:
                    plt.text(v + 0.1, i, f"{v:.1f}%", va='center')
            
            # Save the figure
            plt.savefig(vis_dir / "resource_requirements.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    # 4. Memory Limitations Visualization
    memory_results = benchmark_results.get("memory_limitations", {})
    if "error" not in memory_results:
        plt.figure(figsize=(10, 6))
        
        # Chain coverage
        labels = ['Threats in Chains', 'Isolated Threats']
        chain_coverage = memory_results.get('chain_threat_coverage', 0)
        sizes = [chain_coverage, 1 - chain_coverage]
        
        plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#2ecc71', '#e74c3c']
        )
        plt.axis('equal')
        plt.title('Threat Connectivity in Attack Chains')
        
        # Save the figure
        plt.savefig(vis_dir / "memory_limitations.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 5. Overall Limitations Radar Chart
    plt.figure(figsize=(10, 8))
    
    # Prepare the radar chart
    categories = ['Context Window\nLimitations', 'Probabilistic\nCorrelation', 'Memory\nLimitations']
    values = [
        context_results.get('limitation_score', 0),
        prob_results.get('inconsistency_rate', 0),
        memory_results.get('memory_limitation_score', 0)
    ]
    
    # Close the polygon
    values.append(values[0])
    categories.append(categories[0])
    
    # Convert to radians and calculate points
    angles = np.linspace(0, 2*np.pi, len(categories)-1, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories[:-1], size=12)
    
    # Plot the values
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    
    # Set y-ticks
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["Minimal", "Moderate", "Significant"], color="grey", size=10)
    plt.ylim(0, 1)
    
    plt.title('LLM Limitations in EDR Analysis', size=15, y=1.1)
    
    # Save the figure
    plt.savefig(vis_dir / "overall_limitations.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logging.info(f"Benchmark visualizations saved to {vis_dir}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load results
        results_dir = Path(args.results)
        results = load_results(results_dir)
        
        # Load ground truth if available
        ground_truth = None
        if args.ground_truth:
            ground_truth = load_ground_truth(Path(args.ground_truth))
        
        # Run benchmarks
        benchmark_results = {
            "context_window": analyze_context_window_limitations(results),
            "probabilistic_correlation": analyze_probabilistic_correlation(results, ground_truth),
            "resource_requirements": analyze_resource_requirements(results),
            "memory_limitations": analyze_memory_limitations(results)
        }
        
        # Save benchmark results
        benchmark_output = output_dir / "benchmark_results.json"
        with open(benchmark_output, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logging.info(f"Benchmark results saved to {benchmark_output}")
        
        # Generate report
        report_path = generate_report(benchmark_results, output_dir)
        
        # Generate visualizations
        generate_visualizations(benchmark_results, output_dir)
        
        logging.info(f"Benchmark completed successfully. Report available at {report_path}")
        return 0
        
    except Exception as e:
        logging.error(f"Error during benchmarking: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
