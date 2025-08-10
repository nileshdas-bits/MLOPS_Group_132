"""
Configuration management for the MLOps pipeline
"""
import os
from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """Configuration manager for the MLOps pipeline"""

    def __init__(self):
        self.base_dir  =  Path(__file__).parent.parent.parent
        self.config_file  =  self.base_dir / "config.json"
        self.load_config()

    def load_config(self):
        """Load configuration from file or environment variables"""
        self.config  =  {
            # Data settings
            "data_dir": os.getenv("DATA_DIR", "data"),
            "test_size": float(os.getenv("TEST_SIZE", "0.2")),
            "random_state": int(os.getenv("RANDOM_STATE", "42")),

            # Model settings
            "model_dir": os.getenv("MODEL_DIR", "models"),
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "iris_classification"),

            # API settings
            "api_host": os.getenv("API_HOST", "0.0.0.0"),
            "api_port": int(os.getenv("API_PORT", "8000")),
            "api_reload": os.getenv("API_RELOAD", "True").lower()  ==  "true",

            # MLflow settings
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            "mlflow_registry_uri": os.getenv("MLFLOW_REGISTRY_URI", "file:./mlruns"),

            # Logging settings
            "log_dir": os.getenv("LOG_DIR", "logs"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),

            # Monitoring settings
            "enable_metrics": os.getenv("ENABLE_METRICS", "True").lower()  ==  "true",
            "metrics_port": int(os.getenv("METRICS_PORT", "8000")),

            # Docker settings
            "docker_image": os.getenv("DOCKER_IMAGE", "iris-mlops"),
            "docker_tag": os.getenv("DOCKER_TAG", "latest"),
        }

        # Load from config file if it exists
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                file_config  =  json.load(f)
                self.config.update(file_config)

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent = 2)

    def get(self, key: str, default: Any  =  None) - >  Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key]  =  value

    def get_all(self) - >  Dict[str, Any]:
        """Get all configuration values"""
        return self.config.copy()

    def validate(self) - >  bool:
        """Validate configuration"""
        required_dirs  =  ["data_dir", "model_dir", "log_dir"]

        for dir_key in required_dirs:
            dir_path  =  Path(self.config[dir_key])
            if not dir_path.exists():
                dir_path.mkdir(parents = True, exist_ok = True)

        return True


# Global configuration instance
config  =  Config()