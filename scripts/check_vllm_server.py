#!/usr/bin/env python3
"""
Script to check if the vLLM server is properly configured for Deepseek.
"""

import argparse
import requests
import json
import sys

def check_server(host="localhost", port=8000):
    """
    Check if the vLLM server is properly configured.
    """
    print(f"Checking vLLM server at http://{host}:{port}")
    
    # Try different endpoints
    endpoints = [
        "",  # Root endpoint
        "/v1",
        "/v1/completions",
        "/v1/chat/completions",
        "/generate",
        "/api/v1/generate"
    ]
    
    working_endpoints = []
    
    for endpoint in endpoints:
        url = f"http://{host}:{port}{endpoint}"
        print(f"\nTesting endpoint: {url}")
        
        try:
            # First try a GET request
            response = requests.get(url, timeout=5)
            print(f"GET status: {response.status_code}")
            
            if response.status_code < 400:
                print(f"Endpoint {endpoint} is accessible")
                working_endpoints.append(endpoint)
                
                # Try to print response info
                try:
                    print(f"Response: {response.text[:100]}...")
                except:
                    print("Could not parse response")
            
            # Try a simple inference request
            if endpoint in ["", "/v1", "/v1/chat/completions", "/generate"]:
                print(f"\nTrying inference at {url}")
                
                # Prepare payloads for different endpoints
                if endpoint == "/v1/chat/completions":
                    payload = {
                        "model": "/home/akulg2/edr-llm-analyzer/models/mistral-7b-instruct",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Say hello!"}
                        ],
                        "max_tokens": 10
                    }
                elif endpoint in ["", "/v1", "/generate"]:
                    payload = {
                        "prompt": "Say hello!",
                        "max_tokens": 10
                    }
                
                try:
                    response = requests.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=10
                    )
                    
                    print(f"POST status: {response.status_code}")
                    
                    if response.status_code < 400:
                        print(f"Inference successful at {endpoint}")
                        print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
                        working_endpoints.append(f"{endpoint} (inference)")
                except Exception as e:
                    print(f"Inference failed: {e}")
            
        except Exception as e:
            print(f"Error testing {url}: {e}")
    
    print("\n--- SUMMARY ---")
    if working_endpoints:
        print(f"Working endpoints found: {working_endpoints}")
        print("\nRECOMMENDED CONFIGURATION:")
        
        # Determine best endpoint
        if "/v1/chat/completions (inference)" in working_endpoints:
            print("api_base = \"http://localhost:8000/v1\"")
            print("model = \"mistralai/Mistral-7B-Instruct-v0.3\" (or your actual model name)")
        elif "/v1/chat/completions" in working_endpoints:
            print("api_base = \"http://localhost:8000/v1\"")
            print("model = \"mistralai/Mistral-7B-Instruct-v0.3\" (or your actual model name)")
        elif "/generate (inference)" in working_endpoints:
            print("api_base = \"http://localhost:8000\"")
            print("api_path = \"/generate\"")
            print("model = \"mistralai/Mistral-7B-Instruct-v0.3\" (or your actual model name)")
        else:
            print("No inference endpoints found working, check your server configuration")
    else:
        print("No working endpoints found. Is the vLLM server running?")
        print("You can start the server with:")
        print("""python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --host 0.0.0.0 \\
        --port 8000 \\
        """)
    return 0 if working_endpoints else 1

def main():
    parser = argparse.ArgumentParser(description="Check vLLM server configuration")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    return check_server(args.host, args.port)

if __name__ == "__main__":
    sys.exit(main())
