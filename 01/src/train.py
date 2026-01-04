"""Train a classifier on the processed heart disease data and log with MLflow."""
from __future__ import annotations

import argparse
import joblib
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def train(data_path: Path, model_dir: Path, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.set_experiment('heart-disease')
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        mlflow.log_metric('accuracy', float(acc))
        mlflow.log_metric('roc_auc', float(auc))
        mlflow.sklearn.log_model(clf, artifact_path='model')

        # Save a copy to models/
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_dir / 'rf_heart.joblib')

    return {'accuracy': acc, 'roc_auc': auc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model-dir', required=True)
    args = p.parse_args()
    res = train(Path(args.data), Path(args.model_dir))
    print('metrics:', res)


if __name__ == '__main__':
    main()
