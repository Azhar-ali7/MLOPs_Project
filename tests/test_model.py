"""Tests for model wrapper."""
import pytest
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.model import ModelWrapper


@pytest.fixture
def sample_model(tmp_path):
    """Create a sample trained model."""
    X = pd.DataFrame({
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
        'thal': [3, 2, 2, 3, 3]
    })
    y = [0, 1, 0, 0, 1]
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    
    return model_path


def test_model_wrapper_loads_model(sample_model):
    """Test that ModelWrapper loads the model correctly."""
    wrapper = ModelWrapper(sample_model)
    model = wrapper.load()
    
    assert model is not None
    assert isinstance(model, RandomForestClassifier)


def test_model_wrapper_predict_returns_correct_format(sample_model):
    """Test that predict returns the expected format."""
    wrapper = ModelWrapper(sample_model)
    
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
    
    predictions = wrapper.predict(test_data)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert 'prediction' in predictions[0]
    assert 'probability' in predictions[0]
    assert predictions[0]['prediction'] in [0, 1]
    assert 0 <= predictions[0]['probability'] <= 1


def test_model_wrapper_handles_multiple_rows(sample_model):
    """Test that wrapper handles multiple predictions."""
    wrapper = ModelWrapper(sample_model)
    
    test_data = pd.DataFrame({
        'age': [55, 60, 45],
        'sex': [1, 0, 1],
        'cp': [2, 3, 1],
        'trestbps': [135, 140, 120],
        'chol': [245, 250, 200],
        'fbs': [0, 1, 0],
        'restecg': [0, 0, 1],
        'thalach': [155, 145, 170],
        'exang': [0, 1, 0],
        'oldpeak': [1.5, 2.0, 0.5],
        'slope': [1, 2, 0],
        'ca': [0, 1, 0],
        'thal': [2, 3, 2]
    })
    
    predictions = wrapper.predict(test_data)
    
    assert len(predictions) == 3
    for pred in predictions:
        assert 'prediction' in pred
        assert 'probability' in pred


def test_model_wrapper_raises_on_missing_features(sample_model):
    """Test that wrapper raises error for missing features."""
    wrapper = ModelWrapper(sample_model)
    
    # Missing 'age' column
    test_data = pd.DataFrame({
        'sex': [1],
        'cp': [2],
    })
    
    with pytest.raises(ValueError, match="Feature names mismatch"):
        wrapper.predict(test_data)


def test_model_wrapper_caches_loaded_model(sample_model):
    """Test that model is loaded only once."""
    wrapper = ModelWrapper(sample_model)
    
    model1 = wrapper.load()
    model2 = wrapper.load()
    
    assert model1 is model2  # Same object instance

