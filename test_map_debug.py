#!/usr/bin/env python3
"""
Debug script to test map generation functionality
"""
import requests
import json
import time
import sys

def test_server_connection():
    """Test if server is running"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/test_map_module', timeout=10)
        print(f"Server connection test: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Server connection failed: {e}")
        return False

def test_map_generation():
    """Test map generation with timeout"""
    print("\n=== Testing Map Generation ===")
    
    payload = {
        'files': ['uploads/20250528_1535_20250528_6237_PSALOGV2.mdf'],
        'preset': 'ci_engine_default',
        'min_samples_per_bin': 3
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        print("Sending request...")
        start_time = time.time()
        
        # Set a reasonable timeout
        response = requests.post(
            'http://127.0.0.1:5000/api/compute_map', 
            json=payload,
            timeout=600  # 10 minutes timeout
        )
        
        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.2f} seconds")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            
            if 'meta' in result:
                meta = result['meta']
                print(f"Meta keys: {list(meta.keys())}")
                if 'progress' in meta:
                    progress = meta['progress']
                    print(f"Progress: {progress.get('overall_progress', 'N/A')}%")
                    print(f"Processing time: {progress.get('processing_time', 'N/A')}s")
            
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_file_validation():
    """Test file validation"""
    print("\n=== Testing File Validation ===")
    
    payload = {
        'files': ['uploads/20250528_1535_20250528_6237_PSALOGV2.mdf']
    }
    
    try:
        response = requests.post(
            'http://127.0.0.1:5000/api/validate_files',
            json=payload,
            timeout=30
        )
        
        print(f"Validation status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            validation = result.get('validation', {})
            print(f"Overall score: {validation.get('overall_score', 'N/A')}")
            print(f"Files validated: {len(validation.get('files', {}))}")
            return True
        else:
            print(f"Validation error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Map Generation Debug Test ===")
    
    # Test 1: Server connection
    if not test_server_connection():
        print("❌ Server connection failed")
        sys.exit(1)
    
    # Test 2: File validation
    if not test_file_validation():
        print("❌ File validation failed")
        sys.exit(1)
    
    # Test 3: Map generation
    if not test_map_generation():
        print("❌ Map generation failed")
        sys.exit(1)
    
    print("\n✅ All tests passed!")
