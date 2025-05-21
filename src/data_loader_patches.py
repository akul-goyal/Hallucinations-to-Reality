"""
Patches for the data_loader.py module to handle your unique EDR log format.
This module contains modifications that should be applied to data_loader.py
to better handle the 80GB dataset.
"""
import json
import logging
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

def detect_log_format(file_path: str, sample_size: int = 10000) -> Tuple[str, Dict[str, Any]]:
    """
    Detect the format of EDR log file by examining a sample of the content.
    
    Args:
        file_path: Path to the log file
        sample_size: Size of the sample to read (in bytes)
        
    Returns:
        Tuple of (format, metadata) where format is one of 'json', 'csv', 'syslog', etc.
        and metadata is a dictionary of detected format properties
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Detecting format of: {file_path}")
    
    # Read the first part of the file
    with open(file_path, 'r', errors='ignore') as f:
        sample = f.read(sample_size)
    
    # Check if file appears to be JSON
    if sample.strip().startswith('{') or sample.strip().startswith('['):
        # Try to parse as JSON to confirm and get structure
        try:
            if sample.strip().startswith('['):
                return 'json-array', {'structure': 'array'}
            elif '"events"' in sample and '"events"' in sample[:1000]:
                return 'json-events', {'structure': 'events-array'}
            else:
                return 'json-object', {'structure': 'object'}
        except json.JSONDecodeError:
            pass  # Not valid JSON in sample
    
    # Check if file appears to be CSV
    if ',' in sample and '\n' in sample:
        first_line = sample.split('\n', 1)[0]
        if first_line.count(',') >= 3:  # At least 4 columns
            headers = first_line.split(',')
            # Common EDR headers to look for
            edr_headers = ['timestamp', 'event_type', 'process', 'pid', 'hostname', 'user']
            header_match = sum(1 for h in headers if any(eh in h.lower() for eh in edr_headers))
            if header_match >= 2:  # At least 2 matches
                return 'csv', {'headers': headers}
    
    # Check for syslog format
    syslog_pattern = r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'
    if re.search(syslog_pattern, sample, re.MULTILINE):
        return 'syslog', {'subtype': 'standard'}
    
    # Check for Windows Event Log XML format
    if '<Event xmlns=' in sample or '<EventLog>' in sample:
        return 'xml-eventlog', {'subtype': 'windows'}
    
    # Check for custom format based on common patterns in your logs
    # Custom format detection for your unique EDR format
    if '"telemetry"' in sample or '"telemetry_events"' in sample:
        return 'custom-telemetry', {'structure': 'telemetry-events'}
        
    # Check for JSONL format (JSON Lines - one JSON object per line)
    if '\n' in sample:
        lines = sample.split('\n')
        if any(line.strip().startswith('{') and line.strip().endswith('}') for line in lines[:10]):
            return 'jsonl', {'structure': 'line-per-event'}
    
    # Check for Elastic Common Schema (ECS) format
    if '"@timestamp"' in sample and ('"event"' in sample or '"host"' in sample):
        return 'elastic-ecs', {'structure': 'ecs-format'}
    
    # Check for LEEF (Log Event Extended Format) used by some SIEM systems
    if 'LEEF:' in sample:
        return 'leef', {'subtype': 'siem-format'}
    
    # Check for CEF (Common Event Format)
    if 'CEF:' in sample:
        return 'cef', {'subtype': 'siem-format'}
    
    # Default to unknown format
    return 'unknown', {'sample': sample[:200]}

def parse_custom_telemetry(content: str) -> List[Dict[str, Any]]:
    """
    Parse the custom telemetry format specific to your EDR logs.
    
    Args:
        content: String content of the log file or chunk
        
    Returns:
        List of dictionaries, each representing an EDR event
    """
    events = []
    try:
        data = json.loads(content)
        
        # Handle different possible structures
        if isinstance(data, list):
            # Array of events
            for event in data:
                events.append(normalize_telemetry_event(event))
        elif isinstance(data, dict):
            if "telemetry" in data:
                # Telemetry wrapper
                for event in data["telemetry"]:
                    events.append(normalize_telemetry_event(event))
            elif "telemetry_events" in data:
                # Telemetry events wrapper
                for event in data["telemetry_events"]:
                    events.append(normalize_telemetry_event(event))
            elif "events" in data:
                # Standard events wrapper
                for event in data["events"]:
                    events.append(normalize_telemetry_event(event))
            else:
                # Single event
                events.append(normalize_telemetry_event(data))
    except Exception as e:
        logging.error(f"Error parsing custom telemetry format: {e}")
    
    return events

def normalize_telemetry_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a telemetry event to standard fields for analysis.
    
    Args:
        event: Dictionary containing event data
        
    Returns:
        Normalized event dictionary
    """
    normalized = {}
    
    # Copy the original event
    normalized.update(event)
    
    # Map common fields to standard names
    field_mappings = {
        # Time fields
        "timestamp": ["timestamp", "time", "@timestamp", "event_time", "timeGenerated"],
        "event_type": ["event_type", "type", "eventType", "category", "event.type"],
        "process_name": ["process_name", "process", "image", "ImageName", "process.name", "ProcessName"],
        "process_id": ["process_id", "pid", "ProcessId", "process.pid"],
        "hostname": ["hostname", "host", "Computer", "device_name", "agent.hostname"],
        "username": ["username", "user", "User", "user.name", "UserName"],
        "command_line": ["command_line", "commandline", "CommandLine", "process.command_line"],
        "parent_process_name": ["parent_process_name", "parent_name", "ParentProcessName", "parent.name"],
        "parent_process_id": ["parent_process_id", "parent_pid", "ppid", "ParentProcessId", "parent.pid"],
        "severity": ["severity", "alert_severity", "Severity", "severity_level"],
    }
    
    # Normalize field names
    for standard_field, possible_names in field_mappings.items():
        for name in possible_names:
            if name in event:
                normalized[standard_field] = event[name]
                break
    
    # Ensure event_id exists
    if "event_id" not in normalized:
        if "id" in event:
            normalized["event_id"] = event["id"]
        elif "EventID" in event:
            normalized["event_id"] = event["EventID"]
        elif "_id" in event:
            normalized["event_id"] = event["_id"]
    
    # Handle nested fields (like in Elastic format)
    if "event" in event and isinstance(event["event"], dict):
        if "type" in event["event"] and "event_type" not in normalized:
            normalized["event_type"] = event["event"]["type"]
            
    if "process" in event and isinstance(event["process"], dict):
        if "name" in event["process"] and "process_name" not in normalized:
            normalized["process_name"] = event["process"]["name"]
        if "pid" in event["process"] and "process_id" not in normalized:
            normalized["process_id"] = event["process"]["pid"]
        if "command_line" in event["process"] and "command_line" not in normalized:
            normalized["command_line"] = event["process"]["command_line"]
    
    # Convert severity to numeric if it's a string
    if "severity" in normalized and isinstance(normalized["severity"], str):
        severity_map = {
            "low": 1, "medium": 2, "high": 3, "critical": 4,
            "info": 1, "warning": 2, "error": 3, "fatal": 4,
            "informational": 1, "important": 2, "urgent": 3, "emergency": 4,
            "1": 1, "2": 2, "3": 3, "4": 4
        }
        normalized["severity"] = severity_map.get(normalized["severity"].lower(), 1)
    
    return normalized

def parse_syslog_format(content: str) -> List[Dict[str, Any]]:
    """
    Parse syslog format into structured events.
    
    Args:
        content: String content of the log file or chunk
        
    Returns:
        List of dictionaries, each representing an EDR event
    """
    events = []
    
    # Basic syslog pattern
    syslog_pattern = r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:]+):\s+(.*)$'
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(syslog_pattern, line)
        if match:
            timestamp, hostname, program, message = match.groups()
            
            # Try to extract structured data from message
            event = {
                "timestamp": timestamp,
                "hostname": hostname,
                "program": program,
                "message": message,
                "event_type": "syslog"
            }
            
            # Check if message contains key=value pairs
            if ' = ' in message or '=' in message:
                # Extract key-value pairs
                pairs = re.findall(r'([^=\s]+)\s*=\s*"([^"]*)"', message)
                if not pairs:
                    pairs = re.findall(r'([^=\s]+)\s*=\s*(\S+)', message)
                
                for key, value in pairs:
                    event[key.lower()] = value
            
            events.append(event)
    
    return events

def parse_leef_format(content: str) -> List[Dict[str, Any]]:
    """
    Parse LEEF (Log Event Extended Format) into structured events.
    
    Args:
        content: String content of the log file or chunk
        
    Returns:
        List of dictionaries, each representing an EDR event
    """
    events = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or not line.startswith('LEEF:'):
            continue
        
        # LEEF:Version|Vendor|Product|Version|EventID|key1=value1\tkey2=value2
        parts = line[5:].split('|', 5)
        if len(parts) < 5:
            continue
            
        event = {
            "leef_version": parts[0],
            "vendor": parts[1],
            "product": parts[2],
            "version": parts[3],
            "event_id": parts[4],
            "event_type": "leef"
        }
        
        # Parse the attribute pairs if there are any
        if len(parts) > 5:
            attributes = parts[5]
            # LEEF attributes can be tab-separated or space-separated
            pairs = re.findall(r'([^=\s]+)=(\S+)(?:\s|$)', attributes)
            for key, value in pairs:
                event[key.lower()] = value
        
        events.append(event)
    
    return events

def parse_cef_format(content: str) -> List[Dict[str, Any]]:
    """
    Parse CEF (Common Event Format) into structured events.
    
    Args:
        content: String content of the log file or chunk
        
    Returns:
        List of dictionaries, each representing an EDR event
    """
    events = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or not line.startswith('CEF:'):
            continue
        
        # CEF:Version|Vendor|Product|Version|Signature ID|Name|Severity|Extension
        parts = line[4:].split('|', 7)
        if len(parts) < 7:
            continue
            
        event = {
            "cef_version": parts[0],
            "vendor": parts[1],
            "product": parts[2],
            "version": parts[3],
            "signature_id": parts[4],
            "name": parts[5],
            "severity": parts[6],
            "event_type": "cef"
        }
        
        # Parse the extension
        if len(parts) > 7:
            extension = parts[7]
            # CEF extensions use space as a delimiter with key=value format
            # But values can contain spaces if they're preceded by an escape character
            pairs = re.findall(r'([^=\s]+)=(\S+)(?:\s|$)', extension)
            for key, value in pairs:
                event[key.lower()] = value
        
        events.append(event)
    
    return events

def parse_xml_eventlog(content: str) -> List[Dict[str, Any]]:
    """
    Parse Windows Event Log XML format into structured events.
    
    Args:
        content: String content of the log file or chunk
        
    Returns:
        List of dictionaries, each representing an EDR event
    """
    events = []
    
    try:
        import xml.etree.ElementTree as ET
        
        # Handle both full file and individual events
        if content.strip().startswith('<?xml'):
            # Full XML file
            root = ET.fromstring(content)
            event_elements = root.findall('.//Event')
        else:
            # Individual events
            content_wrapped = f"<Events>{content}</Events>"
            root = ET.fromstring(content_wrapped)
            event_elements = root.findall('./Event')
        
        for event_elem in event_elements:
            event = {"event_type": "windows_event"}
            
            # Extract system metadata
            system_elem = event_elem.find('./System')
            if system_elem is not None:
                for child in system_elem:
                    tag = child.tag.split('}')[-1]  # Remove namespace if present
                    event[tag.lower()] = child.text
                    
                    # Extract attributes
                    for attr_name, attr_value in child.attrib.items():
                        attr_name = attr_name.split('}')[-1]  # Remove namespace
                        event[f"{tag.lower()}_{attr_name.lower()}"] = attr_value
            
            # Extract event data
            event_data = event_elem.find('./EventData')
            if event_data is not None:
                for data in event_data.findall('./Data'):
                    name = data.attrib.get('Name', f"Data_{len(event)}")
                    event[name.lower()] = data.text
            
            events.append(event)
            
    except Exception as e:
        logging.error(f"Error parsing XML EventLog: {e}")
    
    return events

def load_and_normalize_log(file_path: str, chunk_size: int = 1000000) -> pd.DataFrame:
    """
    Load and normalize EDR logs, handling various formats.
    
    Args:
        file_path: Path to the log file
        chunk_size: Number of bytes to read at a time for large files
        
    Returns:
        DataFrame containing normalized EDR events
    """
    logger = logging.getLogger(__name__)
    
    # Detect the log format
    log_format, metadata = detect_log_format(file_path)
    logger.info(f"Detected log format: {log_format}, metadata: {metadata}")
    
    # Load based on format
    events = []
    
    try:
        if log_format == 'unknown':
            logger.warning(f"Unknown log format. Sample: {metadata.get('sample', '')}")
            raise ValueError(f"Unsupported log format detected in {file_path}")
            
        elif log_format.startswith('json'):
            # For JSON formats, we'll use the large_data_processor module
            logger.info(f"Using large_data_processor for JSON format: {log_format}")
            return None  # Signal to use the large_data_processor
            
        elif log_format == 'csv':
            # For CSV, return None to use pandas in the main processor
            logger.info(f"Using pandas for CSV format")
            return None
            
        elif log_format == 'syslog':
            # Read file in chunks due to potential size
            with open(file_path, 'r', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    events.extend(parse_syslog_format(chunk))
                    
        elif log_format == 'leef':
            # Read file in chunks
            with open(file_path, 'r', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    events.extend(parse_leef_format(chunk))
                    
        elif log_format == 'cef':
            # Read file in chunks
            with open(file_path, 'r', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    events.extend(parse_cef_format(chunk))
                    
        elif log_format == 'xml-eventlog':
            # XML files can be large, so process in chunks
            with open(file_path, 'r', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    events.extend(parse_xml_eventlog(chunk))
                    
        elif log_format == 'custom-telemetry':
            # Use the custom telemetry parser
            with open(file_path, 'r', errors='ignore') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    events.extend(parse_custom_telemetry(chunk))
        
        # Convert to DataFrame
        if events:
            logger.info(f"Loaded {len(events)} events from {file_path}")
            df = pd.DataFrame(events)
            
            # Normalize timestamp if present
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                except:
                    logger.warning("Could not convert timestamp column to datetime")
            
            return df
        else:
            logger.warning(f"No events could be parsed from {file_path}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading log file {file_path}: {e}")
        raise

def apply_patches_to_data_loader(data_loader_instance):
    """
    Apply the patches to an instance of EDRDataLoader.
    
    Args:
        data_loader_instance: Instance of EDRDataLoader to patch
    """
    # Add the new methods to the instance
    data_loader_instance.detect_log_format = detect_log_format
    data_loader_instance.load_and_normalize_log = load_and_normalize_log
    
    # Extend the _load_from_file method
    original_load_from_file = data_loader_instance._load_from_file
    
    def patched_load_from_file(file_path: str) -> pd.DataFrame:
        """Patched version of _load_from_file that handles custom formats."""
        try:
            # First try to detect the format
            log_format, metadata = detect_log_format(file_path)
            
            # Handle based on detected format
            if log_format in ['json-array', 'json-events', 'json-object']:
                # Use the original method for JSON
                return original_load_from_file(file_path)
            elif log_format == 'csv':
                # Use the original method for CSV
                return original_load_from_file(file_path)
            else:
                # Use our custom loader for other formats
                df = load_and_normalize_log(file_path)
                if df is not None:
                    return data_loader_instance._preprocess_dataframe(df)
                else:
                    # Fall back to original method
                    return original_load_from_file(file_path)
        except Exception as e:
            logging.error(f"Error in patched _load_from_file: {e}")
            # Fall back to original method
            return original_load_from_file(file_path)
    
    # Replace the method
    data_loader_instance._load_from_file = patched_load_from_file
    
    return data_loader_instance
