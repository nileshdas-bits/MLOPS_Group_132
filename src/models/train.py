"""
Model training script with MLflow integration
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import logging
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.loader import IrisDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")


class ModelTrainer:
    """Model trainer with MLflow integration"""
    
    def __init__(self, experiment_name="iris_classification"):
        """
        Initialize the model trainer
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        with mlflow.start_run(run_name="logistic_regression"):
            # Model parameters
            params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log additional metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{class_name}_{metric_name}", value)
            
            logger.info(f"Logistic Regression - Accuracy: {accuracy:.4f}")
            
            return model, accuracy
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model with hyperparameter tuning"""
        logger.info("Training Random Forest...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            
            # Grid search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Log parameters
            mlflow.log_params(best_params)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log feature importance
            feature_importance = dict(zip(
                ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                best_model.feature_importances_
            ))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            logger.info(f"Random Forest - Accuracy: {accuracy:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return best_model, accuracy
    
    def train_svm(self, X_train, X_test, y_train, y_test):
        """Train Support Vector Machine model"""
        logger.info("Training SVM...")
        
        with mlflow.start_run(run_name="svm"):
            # Model parameters
            params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            model = SVC(**params, probability=True)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"SVM - Accuracy: {accuracy:.4f}")
            
            return model, accuracy
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all models and select the best one"""
        logger.info("Starting model training pipeline...")
        
        models = {}
        scores = {}
        
        # Train Logistic Regression
        lr_model, lr_score = self.train_logistic_regression(X_train, X_test, y_train, y_test)
        models['logistic_regression'] = lr_model
        scores['logistic_regression'] = lr_score
        
        # Train Random Forest
        rf_model, rf_score = self.train_random_forest(X_train, X_test, y_train, y_test)
        models['random_forest'] = rf_model
        scores['random_forest'] = rf_score
        
        # Train SVM
        svm_model, svm_score = self.train_svm(X_train, X_test, y_train, y_test)
        models['svm'] = svm_model
        scores['svm'] = svm_score
        
        # Select best model
        best_model_name = max(scores, key=scores.get)
        self.best_model = models[best_model_name]
        self.best_score = scores[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name} with accuracy: {self.best_score:.4f}")
        
        # Save best model
        self.save_best_model()
        
        return models, scores
    
    def save_best_model(self, model_dir="models"):
        """Save the best model to disk"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save with MLflow
        with mlflow.start_run(run_name="best_model_registration"):
            mlflow.sklearn.log_model(self.best_model, "best_model")
            
            # Log model info
            model_info = {
                "model_name": self.best_model_name,
                "accuracy": self.best_score,
                "model_type": type(self.best_model).__name__
            }
            
            mlflow.log_dict(model_info, "model_info.json")
            
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
            mlflow.register_model(model_uri, "iris-classifier")
            
            logger.info(f"Best model registered: {model_uri}")
        
        # Also save locally for API use
        import joblib
        joblib.dump(self.best_model, model_path / "best_model.pkl")
        
        # Save model info
        with open(model_path / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Best model saved to {model_path}")
    
    def evaluate_model(self, model, X_test, y_test, target_names):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }


def main():
    """Main training function"""
    logger.info("Starting MLOps training pipeline...")
    
    # Load data
    data_loader = IrisDataLoader()
    X_train, X_test, y_train, y_test, feature_names, target_names = data_loader.load_data()
    
    # Save data
    data_loader.save_data(X_train, X_test, y_train, y_test)
    
    # Train models
    trainer = ModelTrainer()
    models, scores = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for model_name, score in scores.items():
        print(f"{model_name.replace('_', ' ').title()}: {score:.4f}")
    
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Best Accuracy: {trainer.best_score:.4f}")
    
    # Evaluate best model
    print("\n" + "="*50)
    print("BEST MODEL EVALUATION")
    print("="*50)
    evaluation = trainer.evaluate_model(trainer.best_model, X_test, y_test, target_names)
    
    print("Training completed successfully!")
    print("Check MLflow UI for detailed experiment tracking.")


if __name__ == "__main__":
    main() 