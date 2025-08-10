"""
Data loader for Iris dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisDataLoader:
    """Data loader for Iris dataset with preprocessing capabilities"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the data loader
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def load_data(self):
        """
        Load the Iris dataset from sklearn
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
        """
        logger.info("Loading Iris dataset...")
        
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Store feature and target names
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Feature names: {self.feature_names}")
        logger.info(f"Target names: {self.target_names}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names, self.target_names
    
    def save_data(self, X_train, X_test, y_train, y_test, output_dir="data"):
        """
        Save the processed data to CSV files
        
        Args:
            X_train, X_test, y_train, y_test: Processed data
            output_dir (str): Directory to save the data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create DataFrames
        train_df = pd.DataFrame(X_train, columns=self.feature_names)
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test, columns=self.feature_names)
        test_df['target'] = y_test
        
        # Save to CSV
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        logger.info(f"Data saved to {output_path}")
        
    def load_saved_data(self, data_dir="data"):
        """
        Load previously saved data
        
        Args:
            data_dir (str): Directory containing the saved data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
        """
        data_path = Path(data_dir)
        
        if not (data_path / "train.csv").exists() or not (data_path / "test.csv").exists():
            logger.warning("Saved data not found. Loading fresh data...")
            return self.load_data()
        
        # Load data
        train_df = pd.read_csv(data_path / "train.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        
        # Separate features and target
        X_train = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values
        X_test = test_df.drop('target', axis=1).values
        y_test = test_df['target'].values
        
        # Get feature names
        self.feature_names = train_df.drop('target', axis=1).columns.tolist()
        self.target_names = ['setosa', 'versicolor', 'virginica']  # Iris target names
        
        logger.info(f"Loaded saved data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test, self.feature_names, self.target_names
    
    def get_sample_data(self):
        """
        Get a sample data point for testing
        
        Returns:
            dict: Sample data point
        """
        sample = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        return sample


def main():
    """Main function to demonstrate data loading"""
    loader = IrisDataLoader()
    
    # Load and save data
    X_train, X_test, y_train, y_test, feature_names, target_names = loader.load_data()
    loader.save_data(X_train, X_test, y_train, y_test)
    
    print("Data loading completed successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Targets: {target_names}")


if __name__ == "__main__":
    main() 