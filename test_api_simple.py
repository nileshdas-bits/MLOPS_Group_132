#!/usr/bin/env python3
"""
Simple API test script
"""
import sys
import os
sys.path.append('src')

from api.main import app
from fastapi.testclient import TestClient

def test_api():
    """Test the API endpoints"""
    client = TestClient(app)
    
    print("Testing API endpoints...")
    
    # Test health check
    print("\n1. Testing health check...")
    response = client.get("/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model_loaded')}")
    
    # Test sample data
    print("\n2. Testing sample data...")
    response = client.get("/sample")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Sample data: {data.get('sample_data')}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data.get('prediction')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Probabilities: {data.get('probabilities')}")
    else:
        print(f"Error: {response.text}")
    
    # Test metrics
    print("\n4. Testing metrics...")
    response = client.get("/metrics")
    print(f"Status: {response.status_code}")
    
    # Test logs
    print("\n5. Testing logs...")
    response = client.get("/logs")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        logs = response.json()
        print(f"Number of logs: {len(logs)}")
    
    print("\nAPI test completed!")

if __name__ == "__main__":
    test_api() 