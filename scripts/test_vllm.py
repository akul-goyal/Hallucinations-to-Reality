#!/usr/bin/env python3
import requests
import json
import time

def test_security_analysis():
    """Test Mistral model on security analysis tasks."""
    url = "http://localhost:8000/v1/chat/completions"
    
    # Security-focused prompt
    security_prompt = """You are a security analyst reviewing EDR logs to identify threats.
    
    Analyze these suspicious events:
    
    Event 1: powershell.exe executed with command: powershell.exe -WindowStyle Hidden -EncodedCommand UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwAgAC0ARgBpAGwAZQBQAGEAdABoACAAQwA6AFwAVABlAG0AcABcAHMAYwByAGkAcAB0AC4AZQB4AGUA
    Event 2: connection to IP 203.0.113.1 over port 443
    Event 3: New file created at C:\\Windows\\Temp\\system_update.exe
    Event 4: system_update.exe executed
    Event 5: rundll32.exe executed with command: rundll32.exe C:\\Windows\\System32\\comsvcs.dll MiniDump 624 C:\\Windows\\Temp\\lsass.dmp full
    
    Analyze these events for potential threats. Return your analysis in JSON format with the following structure:
    {
      "identified_threats": [
        {
          "threat_id": "string",
          "threat_type": "string",
          "related_events": ["string"],
          "severity": "string",
          "confidence": "string",
          "description": "string"
        }
      ],
      "suspicious_patterns": [
        {
          "pattern_id": "string",
          "pattern_type": "string", 
          "related_events": ["string"],
          "description": "string"
        }
      ],
      "risk_assessment": {
        "overall_risk_level": "string",
        "reasoning": "string"
      },
      "recommendations": [
        "string"
      ]
    }"""
    
    payload = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": security_prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Sending security analysis request...")
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    elapsed_time = time.time() - start_time
    
    print(f"Request completed in {elapsed_time:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0]["message"]["content"]
            print("\nAnalysis response:")
            print(message)
            
            # Check if response contains JSON
            try:
                start_idx = message.find('{')
                end_idx = message.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = message[start_idx:end_idx]
                    parsed_json = json.loads(json_str)
                    print("\nSuccessfully parsed JSON response!")
                    
                    # Extract key findings
                    threats = parsed_json.get("identified_threats", [])
                    risk_level = parsed_json.get("risk_assessment", {}).get("overall_risk_level", "Unknown")
                    
                    print(f"\nRisk Assessment: {risk_level}")
                    print(f"Identified {len(threats)} threats:")
                    
                    for i, threat in enumerate(threats):
                        print(f"  {i+1}. {threat.get('threat_type', 'Unknown threat')}: {threat.get('severity', 'Unknown')} severity")
                else:
                    print("\nNo JSON structure found in response")
            except json.JSONDecodeError as e:
                print(f"\nFailed to parse JSON: {e}")
                print("This suggests the model may need additional prompt engineering for structured output")
    else:
        print("Error response:")
        print(response.text)

if __name__ == "__main__":
    test_security_analysis()