"""
Unit tests for the LLM Analyzer module.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest
from pathlib import Path

import sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_analyzer import LLMAnalyzer

class TestLLMAnalyzer(unittest.TestCase):
    """Tests for the LLMAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_config = {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219",
            "api_key": "dummy_key_for_testing",
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        self.analysis_config = {
            "chunk_size": 5,
            "overlap": 1,
            "timeout": 10,
            "max_retries": 1
        }
        
        # Sample EDR data
        self.sample_data = pd.DataFrame([
            {
                "event_id": "evt-1",
                "timestamp": "2025-04-01 12:00:00",
                "event_type": "process",
                "process_name": "powershell.exe",
                "command_line": "powershell.exe -encode ZQBjAGgAbwAgACIASABlAGwAbABvACIAIAA=",
                "hostname": "DESKTOP-ABC123",
                "username": "admin",
                "severity": 3
            },
            {
                "event_id": "evt-2",
                "timestamp": "2025-04-01 12:01:00",
                "event_type": "netconn",
                "process_name": "powershell.exe",
                "remote_ip": "192.168.1.1",
                "remote_port": 443,
                "hostname": "DESKTOP-ABC123",
                "username": "admin",
                "severity": 4
            },
            {
                "event_id": "evt-3",
                "timestamp": "2025-04-01 12:02:00",
                "event_type": "filemod",
                "process_name": "powershell.exe",
                "file_path": "C:\\temp\\suspicious.exe",
                "hostname": "DESKTOP-ABC123",
                "username": "admin",
                "severity": 5
            },
            {
                "event_id": "evt-4",
                "timestamp": "2025-04-01 12:03:00",
                "event_type": "process",
                "process_name": "suspicious.exe",
                "command_line": "suspicious.exe /quiet",
                "hostname": "DESKTOP-ABC123",
                "username": "admin",
                "severity": 6
            },
            {
                "event_id": "evt-5",
                "timestamp": "2025-04-01 12:04:00",
                "event_type": "netconn",
                "process_name": "suspicious.exe",
                "remote_ip": "203.0.113.1",
                "remote_port": 8080,
                "hostname": "DESKTOP-ABC123",
                "username": "admin",
                "severity": 7
            }
        ])
        
        # Sample LLM response
        self.sample_llm_response = """
I've analyzed the EDR events and found suspicious activity:

```json
{
  "identified_threats": [
    {
      "threat_id": "threat-1",
      "threat_type": "Initial Access",
      "related_events": ["evt-1", "evt-2"],
      "severity": "Medium",
      "confidence": "Medium",
      "description": "PowerShell execution with encoded command followed by network connection."
    },
    {
      "threat_id": "threat-2",
      "threat_type": "Execution",
      "related_events": ["evt-3", "evt-4", "evt-5"],
      "severity": "High",
      "confidence": "High",
      "description": "File created and executed, followed by suspicious outbound connection."
    }
  ],
  "suspicious_patterns": [
    {
      "pattern_id": "pattern-1",
      "pattern_type": "PowerShell Encoded Command",
      "related_events": ["evt-1"],
      "description": "PowerShell launched with encoded command parameter."
    }
  ],
  "risk_assessment": {
    "overall_risk_level": "High",
    "reasoning": "Evidence of suspicious file execution and connection to potential C2 server."
  },
  "recommendations": [
    "Investigate the suspicious.exe file",
    "Block connections to 203.0.113.1"
  ],
  "limitations": [
    "Limited event context available"
  ]
}
"""
        
        # Convert timestamp to datetime
        self.sample_data["timestamp"] = pd.to_datetime(self.sample_data["timestamp"])

    @patch('anthropic.Anthropic')
    def test_init(self, mock_anthropic):
        """Test initialization of LLMAnalyzer."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        
        self.assertEqual(analyzer.model, self.llm_config["model"])
        self.assertEqual(analyzer.chunk_size, self.analysis_config["chunk_size"])
        self.assertEqual(analyzer.overlap, self.analysis_config["overlap"])
        
        # Check that Anthropic client was initialized
        mock_anthropic.assert_called_with(api_key=self.llm_config["api_key"])

    def test_create_chunks(self):
        """Test the _create_chunks method."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        chunks = analyzer._create_chunks(self.sample_data)
        
        # With 5 events, chunk_size=5, overlap=1, we should get 1 chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 5)
        
        # Test with smaller chunk size
        analyzer.chunk_size = 3
        analyzer.overlap = 1
        chunks = analyzer._create_chunks(self.sample_data)
        
        # With 5 events, chunk_size=3, overlap=1, we should get 2 chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 3)
        self.assertEqual(len(chunks[1]), 3)  # 3rd, 4th, 5th events

    def test_format_events_for_llm(self):
        """Test the _format_events_for_llm method."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        formatted_events = analyzer._format_events_for_llm(self.sample_data.iloc[0:2])
        
        # Check that the formatted string contains key information
        self.assertIn("evt-1", formatted_events)
        self.assertIn("powershell.exe", formatted_events)
        self.assertIn("DESKTOP-ABC123", formatted_events)
        self.assertIn("netconn", formatted_events)
        
        # Check that the events are properly separated
        event_sections = formatted_events.split("Event ")
        self.assertEqual(len(event_sections), 3)  # Including empty string before "Event 1"

    def test_create_chunk_prompt(self):
        """Test the _create_chunk_prompt method."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        events_text = analyzer._format_events_for_llm(self.sample_data.iloc[0:2])
        prompt = analyzer._create_chunk_prompt(events_text, 0)
        
        # Check that the prompt contains the expected instructions
        self.assertIn("<instructions>", prompt)
        self.assertIn("You are a security analyst", prompt)
        self.assertIn("identified_threats", prompt)
        self.assertIn("suspicious_patterns", prompt)
        self.assertIn("JSON format", prompt)

    @patch('anthropic.Anthropic')
    def test_call_llm_api(self, mock_anthropic):
        """Test the _call_llm_api method."""
        # Configure the mock
        mock_instance = mock_anthropic.return_value
        mock_messages = mock_instance.messages
        mock_create = mock_messages.create
        
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=self.sample_llm_response)]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 800
        mock_create.return_value = mock_response
        
        # Test the method
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        response = analyzer._call_llm_api("Test prompt")
        
        # Check the results
        self.assertEqual(response, self.sample_llm_response)
        mock_create.assert_called_with(
            model=self.llm_config["model"],
            max_tokens=self.llm_config["max_tokens"],
            temperature=self.llm_config["temperature"],
            messages=[{"role": "user", "content": "Test prompt"}],
            timeout=self.analysis_config["timeout"]
        )
        
        # Check that the metrics were updated
        self.assertEqual(analyzer.api_calls, 1)
        self.assertEqual(analyzer.total_tokens, 1300)  # 500 + 800

    def test_parse_llm_response(self):
        """Test the _parse_llm_response method."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        result = analyzer._parse_llm_response(self.sample_llm_response, self.sample_data)
        
        # Check that the JSON was extracted correctly
        self.assertIn("identified_threats", result)
        self.assertEqual(len(result["identified_threats"]), 2)
        self.assertEqual(result["identified_threats"][0]["threat_id"], "threat-1")
        self.assertEqual(result["risk_assessment"]["overall_risk_level"], "High")

    def test_combine_results(self):
        """Test the _combine_results method."""
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        
        # Create two chunk results
        chunk1_result = {
            "chunk_index": 0,
            "structured_results": {
                "identified_threats": [
                    {
                        "threat_id": "threat-1",
                        "threat_type": "Initial Access",
                        "related_events": ["evt-1", "evt-2"],
                        "severity": "Medium",
                        "confidence": "Medium",
                        "description": "PowerShell encoded command"
                    }
                ],
                "suspicious_patterns": [],
                "recommendations": ["Investigate PowerShell usage"],
                "limitations": ["Limited context"]
            }
        }
        
        chunk2_result = {
            "chunk_index": 1,
            "structured_results": {
                "identified_threats": [
                    {
                        "threat_id": "threat-2",
                        "threat_type": "Execution",
                        "related_events": ["evt-3", "evt-4"],
                        "severity": "High",
                        "confidence": "High",
                        "description": "Suspicious file execution"
                    }
                ],
                "suspicious_patterns": [
                    {
                        "pattern_id": "pattern-1",
                        "pattern_type": "Suspicious File Creation",
                        "related_events": ["evt-3"],
                        "description": "Suspicious file created"
                    }
                ],
                "recommendations": ["Block suspicious file"],
                "limitations": ["Limited telemetry"]
            }
        }
        
        combined = analyzer._combine_results([chunk1_result, chunk2_result])
        
        # Check the combined results
        self.assertEqual(len(combined["identified_threats"]), 2)
        self.assertEqual(len(combined["suspicious_patterns"]), 1)
        self.assertEqual(len(combined["recommendations"]), 2)
        self.assertEqual(len(combined["limitations"]), 2)
        
        # Check that threat source chunk was tracked
        self.assertEqual(combined["identified_threats"][0]["source_chunk"], 0)
        self.assertEqual(combined["identified_threats"][1]["source_chunk"], 1)

    @patch.object(LLMAnalyzer, '_call_llm_api')
    @patch.object(LLMAnalyzer, '_process_chunk')
    def test_analyze(self, mock_process_chunk, mock_call_llm_api):
        """Test the analyze method."""
        # Configure the mocks
        mock_process_chunk.side_effect = [
            {
                "chunk_index": 0,
                "structured_results": {
                    "identified_threats": [
                        {
                            "threat_id": "threat-1",
                            "threat_type": "Initial Access",
                            "related_events": ["evt-1", "evt-2"],
                            "severity": "Medium",
                            "confidence": "Medium",
                            "description": "PowerShell encoded command"
                        }
                    ],
                    "suspicious_patterns": [],
                    "recommendations": ["Investigate PowerShell usage"],
                    "limitations": ["Limited context"]
                }
            }
        ]
        
        mock_call_llm_api.return_value = """
{
  "conclusion": "Suspicious PowerShell activity detected",
  "confidence": "Medium",
  "attack_summary": "Initial access via PowerShell with encoded commands",
  "key_indicators": ["Encoded PowerShell command", "Suspicious network connection"],
  "false_positives": ["Legitimate admin tasks may use encoded commands"],
  "reasoning": "Pattern matches known attack techniques",
  "limitations": ["Context window limits comprehensive analysis"]
}
"""

        # Test the method
        analyzer = LLMAnalyzer(self.llm_config, self.analysis_config)
        result = analyzer.analyze(self.sample_data)
        
        # Check that the process_chunk method was called
        mock_process_chunk.assert_called_once()
        
        # Check that the results contain the expected sections
        self.assertIn("identified_threats", result)
        self.assertIn("attack_timeline", result)
        self.assertIn("potential_attack_chains", result)
        self.assertIn("final_summary", result)
        self.assertIn("performance", result)
        
        # Check that performance metrics were tracked
        self.assertIn("api_calls", result["performance"])
        self.assertIn("total_tokens", result["performance"])
        self.assertIn("duration_seconds", result["performance"])

if __name__ == '__main__':
    unittest.main()