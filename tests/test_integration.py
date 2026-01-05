"""Integration tests for the full pipeline."""
import pytest
import pandas as pd
from pathlib import Path
from src.data import load_and_process
from src.train import train
from src.model import ModelWrapper
import tempfile
import shutil


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_base = Path(tempfile.mkdtemp())
    raw_dir = temp_base / "raw"
    processed_dir = temp_base / "processed"
    model_dir = temp_base / "models"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    
    yield {
        'base': temp_base,
        'raw': raw_dir,
        'processed': processed_dir,
        'models': model_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_base)


def create_sample_raw_data(path):
    """Create a sample raw dataset."""
    # Create sample UCI format data (no headers)
    data = [
        [50, 1, 3, 140, 230, 0, 0, 150, 0, 1.0, 0, 0, 3, 0],
        [60, 0, 2, 130, 250, 0, 1, 140, 1, 2.0, 1, 0, 2, 1],
        [55, 1, 1, 145, 240, 1, 0, 160, 0, 1.5, 1, 1, 2, 2],
        [45, 0, 0, 120, 200, 0, 1, 170, 1, 0.5, 0, 0, 3, 0],
        [70, 1, 3, 150, 260, 1, 0, 130, 1, 2.5, 2, 2, 3, 3],
        [65, 1, 2, 135, 245, 0, 0, 155, 0, 1.2, 1, 0, 2, 1],
        [58, 0, 1, 125, 220, 0, 1, 165, 0, 0.8, 0, 0, 3, 0],
        [52, 1, 3, 142, 235, 1, 0, 152, 1, 1.8, 1, 1, 2, 2],
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, header=False)


def test_full_pipeline_integration(temp_dirs):
    """Test the full pipeline from raw data to prediction."""
    # Step 1: Create raw data
    raw_file = temp_dirs['raw'] / "heart.csv"
    create_sample_raw_data(raw_file)
    assert raw_file.exists()
    
    # Step 2: Process data
    processed_file = temp_dirs['processed'] / "heart_processed.csv"
    df_processed = load_and_process(raw_file)
    df_processed.to_csv(processed_file, index=False)
    
    assert processed_file.exists()
    assert 'target' in df_processed.columns
    assert len(df_processed) == 8
    
    # Step 3: Train models
    train_result = train(processed_file, temp_dirs['models'], cv=2)
    
    assert train_result['best_model'] in ['random_forest', 'logistic_regression']
    assert (temp_dirs['models'] / "random_forest.joblib").exists()
    assert (temp_dirs['models'] / "logistic_regression.joblib").exists()
    
    # Step 4: Load model and make predictions
    best_model_name = train_result['best_model']
    model_path = temp_dirs['models'] / f"{best_model_name}.joblib"
    
    wrapper = ModelWrapper(model_path)
    
    # Create test input
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
    
    assert len(predictions) == 1
    assert 'prediction' in predictions[0]
    assert 'probability' in predictions[0]
    assert predictions[0]['prediction'] in [0, 1]
    assert 0 <= predictions[0]['probability'] <= 1


def test_pipeline_handles_missing_values(temp_dirs):
    """Test that pipeline handles missing values correctly."""
    # Create data with missing values
    raw_file = temp_dirs['raw'] / "heart_missing.csv"
    
    data = [
        [50, 1, 3, 140, 230, 0, 0, 150, 0, 1.0, 0, '?', 3, 0],
        [60, 0, 2, 130, 250, 0, 1, 140, 1, 2.0, 1, 0, '?', 1],
        [55, 1, 1, 145, 240, 1, 0, 160, 0, 1.5, 1, 1, 2, 2],
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(raw_file, index=False, header=False)
    
    # Process should handle missing values
    df_processed = load_and_process(raw_file)
    
    assert df_processed.isnull().sum().sum() == 0  # No missing values after processing

