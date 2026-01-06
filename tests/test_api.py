import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def make_and_save_dummy_model(path="models/rf_heart.joblib"):
    # Create a trivial model and save it
    X = pd.DataFrame(
        {
            "age": [50, 60],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [140, 130],
            "chol": [230, 250],
            "fbs": [0, 0],
            "restecg": [0, 1],
            "thalach": [150, 140],
            "exang": [0, 1],
            "oldpeak": [1.0, 2.0],
            "slope": [0, 1],
            "ca": [0, 0],
            "thal": [3, 2],
        }
    )
    y = [0, 1]
    clf = RandomForestClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)


def test_model_wrapper_predict():
    path = "models/rf_heart.joblib"
    make_and_save_dummy_model(path)
    from src.model import ModelWrapper

    mw = ModelWrapper(path)
    df = pd.DataFrame(
        {
            "age": [55],
            "sex": [1],
            "cp": [3],
            "trestbps": [140],
            "chol": [230],
            "fbs": [0],
            "restecg": [0],
            "thalach": [150],
            "exang": [0],
            "oldpeak": [1.0],
            "slope": [0],
            "ca": [0],
            "thal": [3],
        }
    )
    preds = mw.predict(df)
    assert isinstance(preds, list)
    assert "prediction" in preds[0]
