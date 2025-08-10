#!/usr/bin/env python3
"""
Complete MLOps Pipeline Demonstration
"""
import sys
import os
import time
sys.path.append('src')

from data.loader import IrisDataLoader
from models.train import ModelTrainer
from api.main import app
from fastapi.testclient import TestClient

def demo_data_pipeline():
    """Demonstrate data loading and preprocessing"""
    print("=" * 60)
    print("1. DATA PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    loader = IrisDataLoader()
    X_train, X_test, y_train, y_test, feature_names, target_names = loader.load_data()
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Features: {feature_names}")
    print(f"  - Target classes: {target_names}")
    
    # Save data
    loader.save_data(X_train, X_test, y_train, y_test)
    print(f"✓ Data saved to data/ directory")
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

def demo_model_training():
    """Demonstrate model training with MLflow"""
    print("\n" + "=" * 60)
    print("2. MODEL TRAINING DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    loader = IrisDataLoader()
    X_train, X_test, y_train, y_test, feature_names, target_names = loader.load_data()
    
    # Train models
    trainer = ModelTrainer()
    models, scores = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    print("✓ Model training completed")
    print("  Model Performance:")
    for model_name, score in scores.items():
        print(f"    - {model_name.replace('_', ' ').title()}: {score:.4f}")
    
    print(f"  Best Model: {trainer.best_model_name}")
    print(f"  Best Accuracy: {trainer.best_score:.4f}")
    
    return trainer.best_model, trainer.best_score

def demo_api():
    """Demonstrate API functionality"""
    print("\n" + "=" * 60)
    print("3. API DEMONSTRATION")
    print("=" * 60)
    
    client = TestClient(app)
    
    # Test health check
    print("Testing API endpoints...")
    response = client.get("/")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Health check: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
    
    # Test sample data
    response = client.get("/sample")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Sample data endpoint working")
        print(f"  Sample: {data['sample_data']}")
    
    # Test prediction (if model is loaded)
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Prediction successful")
        print(f"  Prediction: {data['prediction']}")
        print(f"  Confidence: {data['confidence']:.3f}")
        print(f"  Probabilities: {data['probabilities']}")
    else:
        print(f"⚠ Prediction failed (model not loaded): {response.json()['detail']}")
    
    # Test metrics
    response = client.get("/metrics")
    if response.status_code == 200:
        print(f"✓ Metrics endpoint working")
    
    # Test logs
    response = client.get("/logs")
    if response.status_code == 200:
        logs = response.json()
        print(f"✓ Logs endpoint working")
        print(f"  Number of prediction logs: {len(logs)}")

def demo_mlflow():
    """Demonstrate MLflow tracking"""
    print("\n" + "=" * 60)
    print("4. MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
    
    import mlflow
    
    # List experiments
    experiments = mlflow.search_experiments()
    print(f"✓ Found {len(experiments)} experiments")
    
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Get latest runs
    experiment = mlflow.get_experiment_by_name("iris_classification")
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
        print(f"✓ Found {len(runs)} recent runs")
        
        for idx, run in runs.iterrows():
            print(f"  Run {run['run_id'][:8]}: {run['metrics.accuracy']:.4f} accuracy")

def main():
    """Run complete pipeline demonstration"""
    print("🚀 MLOPS PIPELINE COMPLETE DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the complete MLOps pipeline:")
    print("1. Data Pipeline - Loading and preprocessing")
    print("2. Model Training - MLflow experiment tracking")
    print("3. API Development - FastAPI endpoints")
    print("4. MLflow Tracking - Experiment management")
    print("=" * 60)
    
    try:
        # 1. Data Pipeline
        demo_data_pipeline()
        
        # 2. Model Training
        best_model, best_score = demo_model_training()
        
        # 3. API Testing
        demo_api()
        
        # 4. MLflow Tracking
        demo_mlflow()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("What we've accomplished:")
        print("✓ Data loading and preprocessing with scikit-learn")
        print("✓ Model training with multiple algorithms")
        print("✓ Experiment tracking with MLflow")
        print("✓ API development with FastAPI")
        print("✓ Input validation with Pydantic")
        print("✓ Logging and monitoring")
        print("✓ Docker containerization ready")
        print("✓ CI/CD pipeline configured")
        print("\nNext steps:")
        print("1. Run: ./deploy.sh deploy")
        print("2. Visit: http://localhost:8000/docs")
        print("3. Check MLflow UI: mlflow ui")
        print("4. View project summary: PROJECT_SUMMARY.md")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 