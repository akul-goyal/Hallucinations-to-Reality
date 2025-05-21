"""
Visualizer for EDR Log Analysis

This module creates visualizations and reports based on the LLM analysis results.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    """
    Class for creating visualizations and reports from LLM analysis results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Dictionary containing visualization configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        self.timeline_format = config.get("timeline_format", "interactive")
        self.color_scheme = config.get("color_scheme", "viridis")
        self.show_confidence = config.get("show_confidence_scores", True)
        
        # Set Seaborn style
        sns.set_theme(style="whitegrid")
        
        # Define color mappings for severity levels
        self.severity_colors = {
            "Low": "#4575b4",     # Blue
            "Medium": "#fee090",  # Yellow
            "High": "#d73027"     # Red
        }
        
        # Define numeric mapping for severity
        self.severity_map = {"High": 3, "Medium": 2, "Low": 1}
    
    def create_visualizations(self, edr_data: pd.DataFrame, 
                            analysis_results: Dict[str, Any], 
                            output_dir: Path) -> Path:
        """
        Create visualizations and reports from analysis results.
        
        Args:
            edr_data: DataFrame containing EDR events
            analysis_results: Dictionary containing analysis results
            output_dir: Path to output directory
            
        Returns:
            Path to the generated report file
        """
        self.logger.info(f"Creating visualizations in {output_dir}")
        
        # Create output directory for visualizations
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check for errors in analysis results
        if "error" in analysis_results:
            self.logger.warning(f"Analysis results contain error: {analysis_results['error']}")
            self._create_error_report(analysis_results, output_dir, timestamp)
            return output_dir / f"error_report_{timestamp}.md"
        
        # Extract key data
        threats = analysis_results.get("identified_threats", [])
        patterns = analysis_results.get("suspicious_patterns", [])
        timeline = analysis_results.get("attack_timeline", [])
        attack_chains = analysis_results.get("potential_attack_chains", [])
        
        # Create visualizations
        try:
            # Event frequency over time
            if not edr_data.empty and "timestamp" in edr_data.columns:
                time_plot_path = self._plot_event_frequency(edr_data, viz_dir, timestamp)
                self.logger.info(f"Event frequency plot saved to {time_plot_path}")
            
            # Threat severity distribution
            if threats:
                severity_plot_path = self._plot_threat_severity(threats, viz_dir, timestamp)
                self.logger.info(f"Threat severity plot saved to {severity_plot_path}")
            
            # Attack timeline
            if timeline:
                timeline_plot_path = self._plot_attack_timeline(timeline, viz_dir, timestamp)
                self.logger.info(f"Attack timeline plot saved to {timeline_plot_path}")
            
            # Threat confidence distribution
            if threats and self.show_confidence:
                confidence_plot_path = self._plot_confidence_distribution(threats, viz_dir, timestamp)
                self.logger.info(f"Confidence distribution plot saved to {confidence_plot_path}")
            
            # Event type distribution
            if not edr_data.empty and "event_type" in edr_data.columns:
                event_type_path = self._plot_event_type_distribution(edr_data, viz_dir, timestamp)
                self.logger.info(f"Event type distribution plot saved to {event_type_path}")
            
            # Attack chain visualization
            if attack_chains:
                chain_plot_path = self._plot_attack_chains(attack_chains, viz_dir, timestamp)
                self.logger.info(f"Attack chain plot saved to {chain_plot_path}")
        
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}", exc_info=True)
        
        # Generate final report
        report_path = self._generate_report(edr_data, analysis_results, viz_dir, output_dir, timestamp)
        self.logger.info(f"Report generated at {report_path}")
        
        return report_path
    
    def _plot_event_frequency(self, data: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
        """
        Plot event frequency over time.
        
        Args:
            data: DataFrame containing EDR events
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating event frequency plot")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Convert timestamp to datetime if not already
        if data["timestamp"].dtype != "datetime64[ns]":
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        
        # Group by time intervals
        # Determine appropriate time interval based on data span
        time_range = data["timestamp"].max() - data["timestamp"].min()
        if time_range.total_seconds() < 3600:  # Less than an hour
            freq = "1min"
            title = "Event Frequency (per minute)"
        elif time_range.total_seconds() < 86400:  # Less than a day
            freq = "1H"
            title = "Event Frequency (hourly)"
        else:
            freq = "1D"
            title = "Event Frequency (daily)"
        
        # Count events per interval
        event_counts = data.set_index("timestamp").resample(freq).size()
        
        # Plot
        ax = event_counts.plot(kind="line", color="steelblue")
        
        # Add event type as points if available
        if "event_type" in data.columns and "severity" in data.columns:
            # Create a color map for severity
            severity_colormap = {
                1: "blue",
                2: "orange",
                3: "red",
                4: "darkred",
                5: "purple"
            }
            
            # Default to 1 if severity not numeric
            if data["severity"].dtype == "object":
                data["severity_num"] = data["severity"].map(self.severity_map).fillna(1)
            else:
                data["severity_num"] = data["severity"]
            
            # Plot individual events as scatter points
            for severity in sorted(data["severity_num"].unique()):
                severity_data = data[data["severity_num"] == severity]
                ax.scatter(
                    severity_data["timestamp"],
                    [1] * len(severity_data),
                    alpha=0.7,
                    s=20 + (severity * 5),  # Size based on severity
                    c=severity_colormap.get(severity, "gray"),
                    label=f"Severity {severity}"
                )
        
        # Format plot
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Number of Events")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Format x-axis based on time range
        if time_range.total_seconds() < 3600:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif time_range.total_seconds() < 86400:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add legend if we plotted severity points
        if "event_type" in data.columns and "severity" in data.columns:
            plt.legend(title="Event Severity", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # Save plot
        output_path = output_dir / f"event_frequency_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_threat_severity(self, threats: List[Dict[str, Any]], output_dir: Path, timestamp: str) -> Path:
        """
        Plot threat severity distribution.
        
        Args:
            threats: List of threat dictionaries
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating threat severity plot")
        
        # Count threats by severity
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for threat in threats:
            severity = threat.get("severity", "Low")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot
        bars = plt.bar(
            severity_counts.keys(),
            severity_counts.values(),
            color=[self.severity_colors[sev] for sev in severity_counts.keys()]
        )
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom"
            )
        
        # Format plot
        plt.title("Threat Severity Distribution")
        plt.xlabel("Severity Level")
        plt.ylabel("Number of Threats")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max(severity_counts.values()) + 1)  # Add space for labels
        
        # Add data labels
        for i, count in enumerate(severity_counts.values()):
            if count > 0:
                plt.annotate(
                    f"{count} ({count/sum(severity_counts.values())*100:.1f}%)",
                    xy=(i, count),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom"
                )
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"threat_severity_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_attack_timeline(self, timeline: List[Dict[str, Any]], output_dir: Path, timestamp: str) -> Path:
        """
        Plot attack timeline.
        
        Args:
            timeline: List of timeline event dictionaries
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating attack timeline plot")
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Convert timeline to DataFrame for easier manipulation
        df = pd.DataFrame(timeline)
        
        # Convert timestamp to datetime if not already
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
        
        # Create a sequential y-axis position for each event
        df["y_pos"] = range(len(df))
        
        # Create color map based on event types
        event_types = df["event_type"].unique()
        colors = plt.cm.get_cmap(self.color_scheme, len(event_types))
        color_map = {event_type: colors(i) for i, event_type in enumerate(event_types)}
        
        # Plot timeline
        for _, row in df.iterrows():
            # Determine marker size based on severity or importance
            marker_size = 100  # Default size
            if "severity" in row and pd.notna(row["severity"]):
                if isinstance(row["severity"], str):
                    # Convert string severity to number
                    severity_num = self.severity_map.get(row["severity"], 1)
                else:
                    severity_num = row["severity"]
                marker_size = 100 + (severity_num * 50)
            
            # Determine color based on event type
            color = color_map.get(row["event_type"], "gray")
            
            # Plot event as a scatter point
            plt.scatter(
                row["timestamp"] if "timestamp" in row else 0,
                row["y_pos"],
                s=marker_size,
                c=[color],
                alpha=0.7,
                edgecolors="black"
            )
            
            # Add event label
            label_text = f"{row['event_type']}: {row['description'][:50]}"
            if "related_threats" in row and row["related_threats"]:
                label_text += f" (Threats: {', '.join(row['related_threats'][:2])})"
            
            plt.text(
                row["timestamp"] if "timestamp" in row else 0,
                row["y_pos"],
                label_text,
                fontsize=9,
                ha="left" if row["y_pos"] % 2 == 0 else "right",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3")
            )
        
        # Format plot
        plt.title("Attack Timeline")
        plt.xlabel("Time")
        plt.yticks([])  # Hide y-axis ticks
        plt.grid(True, axis="x", linestyle="--", alpha=0.7)
        
        # Format x-axis based on time range
        if "timestamp" in df.columns:
            # Calculate time range
            time_range = df["timestamp"].max() - df["timestamp"].min()
            
            if time_range.total_seconds() < 3600:  # Less than an hour
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            elif time_range.total_seconds() < 86400:  # Less than a day
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            else:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        
        plt.xticks(rotation=45)
        
        # Add legend for event types
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=event_type
            )
            for event_type, color in color_map.items()
        ]
        
        plt.legend(
            handles=legend_elements,
            title="Event Types",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"attack_timeline_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_confidence_distribution(self, threats: List[Dict[str, Any]], output_dir: Path, timestamp: str) -> Path:
        """
        Plot threat confidence distribution.
        
        Args:
            threats: List of threat dictionaries
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating confidence distribution plot")
        
        # Count threats by confidence level
        confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
        for threat in threats:
            confidence = threat.get("confidence", "Low")
            if confidence in confidence_counts:
                confidence_counts[confidence] += 1
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # Green, Orange, Red
        bars = plt.bar(
            confidence_counts.keys(),
            confidence_counts.values(),
            color=colors
        )
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom"
            )
        
        # Format plot
        plt.title("LLM Confidence in Threat Identification")
        plt.xlabel("Confidence Level")
        plt.ylabel("Number of Threats")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max(confidence_counts.values()) + 1)  # Add space for labels
        
        # Add data labels
        total = sum(confidence_counts.values())
        for i, count in enumerate(confidence_counts.values()):
            if count > 0 and total > 0:
                plt.annotate(
                    f"{count} ({count/total*100:.1f}%)",
                    xy=(i, count),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom"
                )
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"confidence_distribution_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_event_type_distribution(self, data: pd.DataFrame, output_dir: Path, timestamp: str) -> Path:
        """
        Plot event type distribution.
        
        Args:
            data: DataFrame containing EDR events
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating event type distribution plot")
        
        # Count events by type
        event_counts = data["event_type"].value_counts()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot
        colors = plt.cm.get_cmap(self.color_scheme, len(event_counts))
        bars = plt.bar(
            event_counts.index,
            event_counts.values,
            color=[colors(i) for i in range(len(event_counts))]
        )
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=9
            )
        
        # Format plot
        plt.title("Event Type Distribution")
        plt.xlabel("Event Type")
        plt.ylabel("Number of Events")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"event_type_distribution_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _plot_attack_chains(self, attack_chains: List[Dict[str, Any]], output_dir: Path, timestamp: str) -> Path:
        """
        Visualize attack chains.
        
        Args:
            attack_chains: List of attack chain dictionaries
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the saved plot
        """
        self.logger.debug("Creating attack chains plot")
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Limit to top 5 chains for readability
        chains_to_plot = attack_chains[:5]
        
        # Set up the plot
        ax = plt.subplot(111)
        
        # Plot each chain
        y_offset = 0
        for i, chain in enumerate(chains_to_plot):
            # Get chain details
            chain_id = chain.get("chain_id", f"Chain {i+1}")
            severity = chain.get("severity", "Low")
            threats = chain.get("threats", [])
            
            # Set colors based on severity
            color = self.severity_colors.get(severity, "gray")
            
            # Plot the chain header
            ax.barh(
                y_offset,
                len(threats),
                height=0.5,
                color=color,
                alpha=0.3,
                label=f"{chain_id} (Severity: {severity})"
            )
            
            # Plot each threat in the chain
            for j, threat in enumerate(threats):
                threat_type = threat.get("threat_type", "Unknown")
                confidence = threat.get("confidence", "Medium")
                
                # Adjust transparency based on confidence
                alpha = 0.9 if confidence == "High" else (0.6 if confidence == "Medium" else 0.3)
                
                # Plot the threat
                ax.barh(
                    y_offset,
                    1,
                    left=j,
                    height=0.5,
                    color=color,
                    alpha=alpha,
                    edgecolor="black"
                )
                
                # Add threat label
                ax.text(
                    j + 0.5,
                    y_offset,
                    threat_type,
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=90 if len(threat_type) > 10 else 0
                )
            
            # Move down for the next chain
            y_offset -= 1
        
        # Format plot
        plt.title("Potential Attack Chains")
        plt.xlabel("Attack Progression (Threats)")
        plt.xlim(0, max([len(chain.get("threats", [])) for chain in chains_to_plot]) + 0.5)
        plt.ylim(y_offset + 0.5, 0.5)
        
        # Remove y-axis ticks and add chain labels
        plt.yticks([])
        for i, chain in enumerate(chains_to_plot):
            chain_id = chain.get("chain_id", f"Chain {i+1}")
            severity = chain.get("severity", "Low")
            threat_count = len(chain.get("threats", []))
            event_count = chain.get("event_count", 0)
            
            plt.text(
                -0.5,
                -i,
                f"{chain_id}\nSeverity: {severity}\nThreats: {threat_count}, Events: {event_count}",
                ha="right",
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3")
            )
        
        plt.grid(True, axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"attack_chains_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def _generate_report(self, data: pd.DataFrame, results: Dict[str, Any], 
                        viz_dir: Path, output_dir: Path, timestamp: str) -> Path:
        """
        Generate a comprehensive report of the analysis.
        
        Args:
            data: DataFrame containing EDR events
            results: Dictionary containing analysis results
            viz_dir: Path to visualization directory
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating final report")
        
        # Get report format from config
        report_format = self.config.get("report_format", "markdown")
        
        if report_format == "markdown":
            return self._generate_markdown_report(data, results, viz_dir, output_dir, timestamp)
        else:
            self.logger.warning(f"Unsupported report format: {report_format}. Using markdown instead.")
            return self._generate_markdown_report(data, results, viz_dir, output_dir, timestamp)
    
    def _generate_markdown_report(self, data: pd.DataFrame, results: Dict[str, Any],
                                viz_dir: Path, output_dir: Path, timestamp: str) -> Path:
        """
        Generate a markdown report of the analysis.
        
        Args:
            data: DataFrame containing EDR events
            results: Dictionary containing analysis results
            viz_dir: Path to visualization directory
            output_dir: Path to output directory
            timestamp: Timestamp string for filename
            
        Returns:
            Path to the generated report
        """
        # Get key components
        threats = results.get("identified_threats", [])
        patterns = results.get("suspicious_patterns", [])
        timeline = results.get("attack_timeline", [])
        attack_chains = results.get("potential_attack_chains", [])
        recommendations = results.get("recommendations", [])
        limitations = results.get("limitations", [])
        final_summary = results.get("final_summary", {})
        performance = results.get("performance", {})
        
        # Get relative path to visualizations
        rel_viz_path = viz_dir.relative_to(output_dir)
        
        # Start building the report
        report = []
        
        # Add header
        report.append("# EDR Log Analysis Report using LLMs")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add executive summary
        report.append("## Executive Summary")
        if final_summary:
            report.append(f"**Conclusion:** {final_summary.get('conclusion', 'No conclusion provided.')}")
            report.append(f"**Confidence:** {final_summary.get('confidence', 'Medium')}")
            
            if "attack_summary" in final_summary:
                report.append(f"\n{final_summary['attack_summary']}")
            
            if "key_indicators" in final_summary and final_summary["key_indicators"]:
                report.append("\n**Key Indicators of Compromise:**")
                for indicator in final_summary["key_indicators"]:
                    report.append(f"- {indicator}")
            
            if "reasoning" in final_summary:
                report.append(f"\n**Reasoning:** {final_summary['reasoning']}")
        else:
            report.append("No summary provided.")
        report.append("")
        
        # Add risk assessment
        report.append("## Risk Assessment")
        risk_assessment = results.get("risk_assessment", {})
        if risk_assessment:
            report.append(f"**Overall Risk Level:** {risk_assessment.get('overall_risk_level', 'Unknown')}")
            report.append(f"**Reasoning:** {risk_assessment.get('reasoning', 'No reasoning provided.')}")
            
            if "severity_distribution" in risk_assessment:
                severity_dist = risk_assessment["severity_distribution"]
                report.append("\n**Severity Distribution:**")
                for severity, count in severity_dist.items():
                    report.append(f"- {severity}: {count}")
        else:
            report.append("No risk assessment provided.")
        report.append("")
        
        # Add identified threats
        report.append("## Identified Threats")
        if threats:
            report.append(f"Found {len(threats)} potential threats:")
            
            # Sort threats by severity
            sorted_threats = sorted(
                threats,
                key=lambda x: self.severity_map.get(x.get("severity", "Low"), 0),
                reverse=True
            )
            
            for i, threat in enumerate(sorted_threats):
                threat_id = threat.get("threat_id", f"Threat-{i+1}")
                threat_type = threat.get("threat_type", "Unknown")
                severity = threat.get("severity", "Low")
                confidence = threat.get("confidence", "Medium")
                description = threat.get("description", "No description provided.")
                
                report.append(f"\n### {threat_id}: {threat_type}")
                report.append(f"**Severity:** {severity}  |  **Confidence:** {confidence}")
                report.append(f"\n{description}")
                
                if "related_events" in threat and threat["related_events"]:
                    report.append("\n**Related Events:**")
                    for event_id in threat["related_events"][:5]:  # Limit to 5 events
                        report.append(f"- {event_id}")
                    
                    if len(threat["related_events"]) > 5:
                        report.append(f"- ... and {len(threat['related_events']) - 5} more")
        else:
            report.append("No threats identified.")
        report.append("")
        
        # Add suspicious patterns
        if patterns:
            report.append("## Suspicious Patterns")
            report.append(f"Found {len(patterns)} suspicious patterns:")
            
            for i, pattern in enumerate(patterns):
                pattern_id = pattern.get("pattern_id", f"Pattern-{i+1}")
                pattern_type = pattern.get("pattern_type", "Unknown")
                description = pattern.get("description", "No description provided.")
                
                report.append(f"\n### {pattern_id}: {pattern_type}")
                report.append(f"\n{description}")
                
                if "related_events" in pattern and pattern["related_events"]:
                    report.append("\n**Related Events:**")
                    for event_id in pattern["related_events"][:5]:  # Limit to 5 events
                        report.append(f"- {event_id}")
                    
                    if len(pattern["related_events"]) > 5:
                        report.append(f"- ... and {len(pattern['related_events']) - 5} more")
            report.append("")
        
        # Add attack chains
        if attack_chains:
            report.append("## Potential Attack Chains")
            
            # Add attack chain visualization if available
            chain_plot_file = list(viz_dir.glob("attack_chains_*.png"))
            if chain_plot_file:
                plot_path = chain_plot_file[0].relative_to(output_dir)
                report.append(f"![Attack Chains]({plot_path})")

        # Add LLM limitations for security analysis
        report.append("## LLM Limitations for Security Analysis")
        report.append("\nThis analysis demonstrates several key limitations of using LLMs for cybersecurity threat analysis:")
        # Get chunk_size from analysis_config if available, otherwise use a default value
        analysis_config = results.get("analysis_config", {})
        self.chunk_size = analysis_config.get("chunk_size", 50)  # Default to 50 if not available

        report.append("\n### 1. Context Window Constraints")
        report.append("The LLM can only analyze a limited number of events at once, potentially missing patterns that span across larger datasets. " 
                     f"This analysis required dividing {len(data)} events into {results.get('chunk_count', 1)} chunks with {self.chunk_size} events per chunk.")

        report.append("\n### 2. Probabilistic Correlation")
        report.append("LLM-based analysis relies on probabilistic pattern matching rather than deterministic rule-based detection. "
                     "This can lead to both false positives and missed connections between related events.")

        report.append("\n### 3. Resource Requirements")
        if results.get("resource_usage"):
            resource_usage = results.get("resource_usage", {})
            memory_usage = resource_usage.get("memory_usage_GB", {})
            max_mem = memory_usage.get("max", "N/A")
            report.append(f"This analysis consumed significant computational resources: peak memory usage of {max_mem:.2f} GB. "
                         f"Processing took {performance.get('duration_seconds', 0):.2f} seconds for {len(data)} events.")
        else:
            report.append("LLM-based analysis requires significant computational resources and API costs, scaling linearly with dataset size.")

        report.append("\n### 4. Lack of Memory")
        report.append("The LLM has no persistent memory between chunks, making it difficult to track complex attack patterns that evolve over time. "
                     "This is particularly problematic for advanced persistent threats (APTs) that operate over extended periods.")

        if limitations:
            report.append("\n### Additional Limitations Identified During Analysis")
            for limitation in limitations:
                report.append(f"- {limitation}")

        report.append("")
