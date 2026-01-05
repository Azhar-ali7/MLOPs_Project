"""Tests for model training pipeline."""
import pytest
import pandas as pd
import joblib
from pathlib import Path
from src.train import train


@pytest.fixture
def sample_data_path(tmp_path):
    """Create a sample dataset for testing."""
    data = pd.DataFrame({
        'age': [50, 60, 55, 45, 70],
        'sex': [1, 0, 1, 0, 1],
        'cp': [3, 2, 1, 0, 3],
        'trestbps': [140, 130, 145, 120, 150],
        'chol': [230, 250, 240, 200, 260],
        'fbs': [0, 0, 1, 0, 1],
        'restecg': [0, 1, 0, 1, 0],
        'thalach': [150, 140, 160, 170, 130],
        'exang': [0, 1, 0, 1, 1],
        'oldpeak': [1.0, 2.0, 1.5, 0.5, 2.5],
        'slope': [0, 1, 1, 0, 2],
        'ca': [0, 0, 1, 0, 2],
        'thal': [3, 2, 2, 3, 3],
        'target': [0, 1, 0, 0, 1]
    })
    
    data_file = tmp_path / "test_data.csv"
    data.to_csv(data_file, index=False)
    return data_file


def test_train_creates_models(sample_data_path, tmp_path):
    """Test that training creates model files."""
    model_dir = tmp_path / "models"
    
    result = train(sample_data_path, model_dir, cv=2)
    
    assert 'best_model' in result
    assert 'best_score' in result
    assert model_dir.exists()
    
    # Check that model files were created
    assert (model_dir / "random_forest.joblib").exists()
    assert (model_dir / "logistic_regression.joblib").exists()
    assert (model_dir / "selection_summary.json").exists()


def test_train_returns_valid_summary(sample_data_path, tmp_path):
    """Test that training returns a valid summary."""
    model_dir = tmp_path / "models"
    
    result = train(sample_data_path, model_dir, cv=2)
    
    assert isinstance(result, dict)
    assert result['best_model'] in ['random_forest', 'logistic_regression']
    assert 0 <= result['best_score'] <= 1
    assert 'results' in result
    
    # Check results structure
    for model_name in ['random_forest', 'logistic_regression']:
        assert model_name in result['results']
        assert 'metrics' in result['results'][model_name]
        assert 'best_params' in result['results'][model_name]


def test_trained_model_can_predict(sample_data_path, tmp_path):
    """Test that trained models can make predictions."""
    model_dir = tmp_path / "models"
    
    train(sample_data_path, model_dir, cv=2)
    
    # Load the best model
    rf_model = joblib.load(model_dir / "random_forest.joblib")
    
    # Create sample input
    test_data = pd.DataFrame({
        'age': [55],
        'sex': [1],
        'cp': [2],
        'trestbps': [135],
        'chol': [245],
        'fbs': [0],
        'restecg': [0],
        'thalach': [155],
        'exang': [0],
        'oldpeak': [1.5],
        'slope': [1],
        'ca': [0],
        'thal': [2]
    })
    
    # Make prediction
    prediction = rf_model.predict(test_data)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

