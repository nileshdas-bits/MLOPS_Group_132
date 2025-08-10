#!/usr/bin/env python3
"""
Test script to verify MLOps pipeline setup
"""
import sys
import os
import subprocess
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import mlflow
        print("✓ mlflow imported successfully")
    except ImportError as e:
        print(f"✗ mlflow import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✓ fastapi imported successfully")
    except ImportError as e:
        print(f"✗ fastapi import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✓ uvicorn imported successfully")
    except ImportError as e:
        print(f"✗ uvicorn import failed: {e}")
        return False
    
    return True

def test_data_loader():
    """Test data loader functionality"""
    print("\nTesting data loader...")
    
    try:
        sys.path.append('src')
        from data.loader import IrisDataLoader
        
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test, feature_names, target_names = loader.load_data()
        
        print(f"✓ Data loaded successfully")
        print(f"  - Training samples: {X_train.shape[0]}")
        print(f"  - Test samples: {X_test.shape[0]}")
        print(f"  - Features: {feature_names}")
        print(f"  - Targets: {target_names}")
        
        return True
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        sys.path.append('src')
        from data.loader import IrisDataLoader
        from models.train import ModelTrainer
        
        # Load data
        loader = IrisDataLoader()
        X_train, X_test, y_train, y_test, feature_names, target_names = loader.load_data()
        
        # Train a simple model
        trainer = ModelTrainer()
        model, accuracy = trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
        
        print(f"✓ Model training successful")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def test_api_import():
    """Test API import"""
    print("\nTesting API import...")
    
    try:
        sys.path.append('src')
        from api.main import app
        
        print("✓ FastAPI app imported successfully")
        print(f"  - App title: {app.title}")
        print(f"  - App version: {app.version}")
        
        return True
    except Exception as e:
        print(f"✗ API import test failed: {e}")
        return False

def test_file_structure():
    """Test that required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "deploy.sh",
        "README.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "src",
        "src/data",
        "src/models", 
        "src/api",
        "src/utils",
        "tests",
        "data",
        "models",
        "logs",
        ".github/workflows"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ missing")
            all_good = False
    
    return all_good

def test_docker():
    """Test Docker functionality"""
    print("\nTesting Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Docker available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker not available or not working")
        return False

def main():
    """Run all tests"""
    print("MLOps Pipeline Setup Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Model Training", test_model_training),
        ("API Import", test_api_import),
        ("Docker", test_docker)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your MLOps pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run: ./deploy.sh deploy")
        print("2. Visit: http://localhost:8000/docs")
        print("3. Check MLflow UI: mlflow ui")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 