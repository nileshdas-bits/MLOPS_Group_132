"""
Unit tests for data loader
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import IrisDataLoader


class TestIrisDataLoader:
    """Test cases for IrisDataLoader"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loader = IrisDataLoader(test_size=0.2, random_state=42)
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        assert self.loader.test_size == 0.2
        assert self.loader.random_state == 42
        assert self.loader.scaler is not None
        assert self.loader.feature_names is None
        assert self.loader.target_names is None
    
    def test_load_data(self):
        """Test data loading"""
        X_train, X_test, y_train, y_test, feature_names, target_names = self.loader.load_data()
        
        # Check data shapes
        assert X_train.shape[1] == 4  # 4 features
        assert X_test.shape[1] == 4
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check feature names
        expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        assert feature_names == expected_features
        
        # Check target names
        expected_targets = ['setosa', 'versicolor', 'virginica']
        assert target_names == expected_targets
        
        # Check that data is scaled
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-10)
    
    def test_save_and_load_data(self):
        """Test saving and loading data"""
        # Load and save data
        X_train, X_test, y_train, y_test, feature_names, target_names = self.loader.load_data()
        self.loader.save_data(X_train, X_test, y_train, y_test, str(self.test_data_dir))
        
        # Check files exist
        assert (self.test_data_dir / "train.csv").exists()
        assert (self.test_data_dir / "test.csv").exists()
        
        # Load saved data
        X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded, _, _ = self.loader.load_saved_data(str(self.test_data_dir))
        
        # Check data consistency
        np.testing.assert_array_equal(X_train, X_train_loaded)
        np.testing.assert_array_equal(X_test, X_test_loaded)
        np.testing.assert_array_equal(y_train, y_train_loaded)
        np.testing.assert_array_equal(y_test, y_test_loaded)
    
    def test_get_sample_data(self):
        """Test sample data generation"""
        sample = self.loader.get_sample_data()
        
        assert isinstance(sample, dict)
        assert 'sepal_length' in sample
        assert 'sepal_width' in sample
        assert 'petal_length' in sample
        assert 'petal_width' in sample
        
        # Check data types
        for value in sample.values():
            assert isinstance(value, (int, float))
            assert value > 0
    
    def test_data_consistency(self):
        """Test that data is consistent across multiple loads"""
        # Load data twice
        X_train1, X_test1, y_train1, y_test1, _, _ = self.loader.load_data()
        X_train2, X_test2, y_train2, y_test2, _, _ = self.loader.load_data()
        
        # Check consistency
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_target_distribution(self):
        """Test that target distribution is maintained"""
        X_train, X_test, y_train, y_test, _, _ = self.loader.load_data()
        
        # Check that all classes are present
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)
        
        assert len(unique_train) == 3  # 3 classes
        assert len(unique_test) == 3
        assert np.array_equal(unique_train, unique_test)
        
        # Check that classes are balanced (approximately)
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        
        # Each class should have at least some samples
        assert np.all(train_counts > 0)
        assert np.all(test_counts > 0)


if __name__ == "__main__":
    pytest.main([__file__]) 