"""
Utility Functions for EDR Log Analysis

This module contains helper functions used across the EDR log analysis project.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up handlers
    handlers = [
        logging.StreamHandler(sys.stdout),  # Console handler
        logging.FileHandler(log_dir / f"edr_analysis_{os.getpid()}.log")  # File handler
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    
    logging.info(f"Logging initialized at level {logging.getLevelName(level)}")

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required top-level sections
    required_sections = ["llm", "edr", "analysis", "output"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate LLM settings
    llm_config = config["llm"]
    if "model" not in llm_config:
        raise ValueError("Missing required LLM model setting")
    
    # If provider is deepseek, validate deepseek config
    if llm_config.get("provider", "").lower() == "deepseek":
        if "deepseek" not in config:
            raise ValueError("Missing Deepseek configuration section when provider is 'deepseek'")
        
        deepseek_config = config["deepseek"]
        if "api_key" not in deepseek_config and "api_key" not in llm_config:
            raise ValueError("Missing API key in Deepseek configuration")
        
        if "model" not in deepseek_config and "model" not in llm_config:
            raise ValueError("Missing model name in Deepseek configuration")
    
    # Validate EDR settings
    edr_config = config["edr"]
    if "source" not in edr_config:
        raise ValueError("Missing required EDR source setting")
        
    # Validate analysis settings
    analysis_config = config["analysis"]
    if "chunk_size" not in analysis_config:
        raise ValueError("Missing required chunk_size setting")
    
    # Check for inconsistent settings
    if analysis_config.get("overlap", 0) >= analysis_config.get("chunk_size", 0):
        raise ValueError("Overlap must be less than chunk_size")
    
    # Validate numeric settings are positive
    numeric_settings = [
        ("llm", "max_tokens"),
        ("edr", "max_events"),
        ("analysis", "chunk_size"),
        ("analysis", "timeout")
    ]
    
    for section, setting in numeric_settings:
        if setting in config[section] and config[section][setting] <= 0:
            raise ValueError(f"{section}.{setting} must be a positive number")

def format_filename(filename: str) -> str:
    """
    Format a filename to be safe for use in file paths.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Trim whitespace and limit length
    filename = filename.strip()
    if len(filename) > 200:
        filename = filename[:197] + "..."
    
    return filename

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{int(minutes)} minutes {int(seconds % 60)} seconds"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)} hours {int(minutes)} minutes"

def save_results(results: Dict[str, Any], output_path: Path) -> Path:
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Analysis results dictionary
        output_path: Directory to save the results
        
    Returns:
        Path to the saved file
    """
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"analysis_results_{timestamp}.json"
    
    # Convert datetime objects to strings
    def json_serializer(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # Save results
    try:
        with open(file_path, 'w') as file:
            json.dump(results, file, default=json_serializer, indent=2)
        
        logging.info(f"Results saved to {file_path}")
        return file_path
    
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise

def generate_sample_data(output_path: Path, num_events: int = 100) -> Path:
    """
    Generate sample EDR data for testing.
    
    Args:
        output_path: Directory to save the sample data
        num_events: Number of events to generate
        
    Returns:
        Path to the saved file
    """
    import json
    import random
    from datetime import datetime, timedelta
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Event types with relative frequencies
    event_types = {
        "process": 30,
        "netconn": 20,
        "filemod": 15,
        "regmod": 10,
        "moduleload": 10,
        "childproc": 8,
        "crossproc": 5,
        "remotethread": 1,
        "tamper": 1
    }
    
    # Weighted event type list
    weighted_types = []
    for event_type, weight in event_types.items():
        weighted_types.extend([event_type] * weight)
    
    # Sample process names
    process_names = [
        "svchost.exe", "explorer.exe", "chrome.exe", "firefox.exe", "outlook.exe",
        "powershell.exe", "cmd.exe", "rundll32.exe", "regsvr32.exe", "services.exe",
        "lsass.exe", "winlogon.exe", "csrss.exe", "conhost.exe", "notepad.exe",
        "taskmgr.exe", "iexplore.exe", "wmiprvse.exe", "spoolsv.exe", "msiexec.exe"
    ]
    
    # Sample hostnames
    hostnames = [
        "DESKTOP-1A2B3C4", "LAPTOP-XYZ1234", "WIN-ABCDEF", "WORKSTATION-01",
        "SERVER-DC-01", "FINANCE-PC-02", "HR-LAPTOP-05", "RECEPTION-PC", "CEO-LAPTOP"
    ]
    
    # Sample usernames
    usernames = [
        "admin", "jsmith", "agarcia", "mwilliams", "rjohnson", "tlee", 
        "system", "NETWORK SERVICE", "LOCAL SERVICE", "bjones"
    ]
    
    # Sample commands
    commands = [
        "C:\\Windows\\System32\\{} /s /p",
        "{} --no-sandbox --user-data-dir=C:\\Users\\{}\\AppData\\Local\\Temp",
        "\"{}\" -WindowStyle Hidden -ExecutionPolicy Bypass -Command {}",
        "{} /c {} & {}",
        "C:\\Program Files\\{}\\{} --update",
        "{} -k {} -o {} -p {}"
    ]
    
    # Sample IPs
    ips = [
        "192.168.1.1", "10.0.0.1", "172.16.0.1", "8.8.8.8", "1.1.1.1",
        "203.0.113.1", "198.51.100.1", "104.16.23.29", "172.217.168.238"
    ]
    
    # Generate events
    events = []
    base_time = datetime.now() - timedelta(hours=24)
    pids = {}  # Keep track of processes
    
    for i in range(num_events):
        # Generate timestamp
        event_time = base_time + timedelta(seconds=i*60 + random.randint(0, 30))
        
        # Select event type
        event_type = random.choice(weighted_types)
        
        # Generate event data
        hostname = random.choice(hostnames)
        username = random.choice(usernames)
        
        # Generate process info
        process_name = random.choice(process_names)
        process_pid = random.randint(1000, 9999)
        
        # Store process info
        pids[process_pid] = process_name
        
        # Generate parent process info
        if pids and random.random() < 0.7:  # 70% chance to have a known parent
            parent_pid = random.choice(list(pids.keys()))
            parent_name = pids[parent_pid]
        else:
            parent_pid = random.randint(1000, 9999)
            parent_name = random.choice(process_names)
            pids[parent_pid] = parent_name
        
        # Generate command line
        if event_type in ["process", "childproc"]:
            command_tpl = random.choice(commands)
            command_line = command_tpl.format(
                process_name,
                random.choice(process_names),
                username,
                random.choice(process_names),
                random.choice(process_names),
                random.choice(process_names)
            )
        else:
            command_line = ""
        
        # Generate severity (1-8)
        severity = 1  # Default low severity
        
        # Higher severity for suspicious activity
        if process_name in ["powershell.exe", "cmd.exe", "rundll32.exe", "regsvr32.exe"]:
            severity += 1
        
        if event_type in ["remotethread", "tamper", "crossproc"]:
            severity += 2
        
        # Generate the event
        event = {
            "event_id": f"evt-{i+1}",
            "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "hostname": hostname,
            "username": username,
            "process_name": process_name,
            "process_pid": process_pid,
            "parent_name": parent_name,
            "parent_id": parent_pid,
            "command_line": command_line,
            "severity": severity
        }
        
        # Add event-type specific fields
        if event_type == "netconn":
            event["remote_ip"] = random.choice(ips)
            event["remote_port"] = random.choice([80, 443, 8080, 22, 3389, 445, 135])
            event["protocol"] = random.choice(["TCP", "UDP"])
            
        elif event_type == "filemod":
            event["file_path"] = "C:\\Users\\" + username + "\\" + random.choice(['Documents', 'Downloads', 'AppData\\Local\\Temp']) + "\\" + str(random.randint(1, 999)) + "." + random.choice(['exe', 'dll', 'bat', 'ps1', 'txt'])
            event["file_operation"] = random.choice(["create", "modify", "delete", "rename"])
            
        elif event_type == "regmod":
            event["registry_key"] = "HKEY_" + random.choice(['LOCAL_MACHINE', 'CURRENT_USER']) + "\\" + random.choice(['SOFTWARE\\Microsoft\\Windows', 'SYSTEM\\CurrentControlSet', 'SOFTWARE\\Policies'])
            event["registry_operation"] = random.choice(["create", "modify", "delete"])
        
        # Add a malicious IOC in 5% of events
        if random.random() < 0.05:
            event["ioc_match"] = True
            event["ioc_type"] = random.choice(["hash", "domain", "ip", "file", "registry"])
            event["severity"] += 2
            
        events.append(event)
    
    # Inject a suspicious sequence for detection
    if num_events >= 10:
        attack_start = random.randint(10, num_events - 10)
        
        # Stage 1: Command and control
        events[attack_start]["event_type"] = "netconn"
        events[attack_start]["process_name"] = "powershell.exe"
        events[attack_start]["remote_ip"] = "203.0.113.1"  # Fictitious IOC
        events[attack_start]["remote_port"] = 443
        events[attack_start]["protocol"] = "TCP"
        events[attack_start]["severity"] = 4
        events[attack_start]["ioc_match"] = True
        
        # Stage 2: Privilege escalation
        events[attack_start + 1]["event_type"] = "process"
        events[attack_start + 1]["process_name"] = "cmd.exe"
        events[attack_start + 1]["command_line"] = "cmd.exe /c whoami /priv"
        events[attack_start + 1]["parent_name"] = "powershell.exe"
        events[attack_start + 1]["parent_id"] = events[attack_start]["process_pid"]
        events[attack_start + 1]["severity"] = 5
        
        # Stage 3: Credential access
        events[attack_start + 2]["event_type"] = "process"
        events[attack_start + 2]["process_name"] = "rundll32.exe"
        events[attack_start + 2]["command_line"] = "rundll32.exe C:\\Windows\\System32\\comsvcs.dll MiniDump 624 C:\\temp\\lsass.dmp full"
        events[attack_start + 2]["parent_name"] = "cmd.exe"
        events[attack_start + 2]["parent_id"] = events[attack_start + 1]["process_pid"]
        events[attack_start + 2]["severity"] = 8
        
        # Stage 4: Data exfiltration
        events[attack_start + 3]["event_type"] = "netconn"
        events[attack_start + 3]["process_name"] = "powershell.exe"
        events[attack_start + 3]["remote_ip"] = "203.0.113.1"  # Same as C2
        events[attack_start + 3]["remote_port"] = 443
        events[attack_start + 3]["protocol"] = "TCP"
        events[attack_start + 3]["severity"] = 7
    
    # Save events to file
    file_path = output_path / "sample_edr_data.json"
    with open(file_path, 'w') as file:
        json.dump({"events": events}, file, indent=2)
    
    logging.info(f"Generated {num_events} sample EDR events at {file_path}")
    return file_path

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse a timestamp string into a datetime object.
    
    Args:
        timestamp_str: Timestamp string in various formats
        
    Returns:
        Datetime object or None if parsing fails
    """
    import pandas as pd
    
    try:
        return pd.to_datetime(timestamp_str)
    except:
        return None

# Import pandas here to avoid circular imports
try:
    import pandas as pd
except ImportError:
    logging.warning("pandas not available. Some functions may not work.")
