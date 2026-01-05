"""Train classifiers (Random Forest and Logistic Regression), perform hyperparameter tuning
and cross-validation evaluation, log experiments to MLflow, and save the best model.

The script runs a GridSearchCV for each model, evaluates the best estimator using cross-validation
metrics (accuracy, precision, recall, ROC-AUC), logs results to MLflow, and saves the chosen
model artifact plus a `features.json` file with the training feature order.
"""
from __future__ import annotations

import argparse
import json
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(data_path: Path, model_dir: Path, cv: int = 5, random_state: int = 42):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    model_dir.mkdir(parents=True, exist_ok=True)

    # Define candidates and hyperparameter grids
    rf = RandomForestClassifier(random_state=random_state)
    lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))])

    candidates = {
        'random_forest': (rf, {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10]
        }),
        'logistic_regression': (lr, {
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        })
    }

    mlflow.set_experiment('heart-disease')

    results = {}
    best_overall = None
    best_score = -float('inf')

    outer_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, (estimator, param_grid) in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param('model', name)
            # Grid search to tune hyperparameters (optimize ROC-AUC)
            gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
            gs.fit(X, y)

            best = gs.best_estimator_
            best_params = gs.best_params_
            mlflow.log_params(best_params)

            # Evaluate best estimator using cross-validated metrics
            scoring = ['accuracy', 'precision', 'recall', 'roc_auc']
            cv_res = cross_validate(best, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1, return_train_score=False)

            metrics_mean = {f'{k}_mean': float(cv_res[f'test_{k}'].mean()) for k in scoring}
            for k, v in metrics_mean.items():
                mlflow.log_metric(k, v)

            # Log the model artifact
            mlflow.sklearn.log_model(best, artifact_path='model')

            # Save local copy and features ordering
            model_path = model_dir / f'{name}.joblib'
            joblib.dump(best, model_path)
            features_path = model_dir / f'{name}_features.json'
            # try to get feature order from estimator attribute, otherwise use X.columns
            feature_order = getattr(getattr(best, 'feature_names_in_', None), 'tolist', lambda: list(X.columns))()
            # If pipeline, extract from last estimator if available
            if hasattr(best, 'named_steps') and 'clf' in getattr(best, 'named_steps'):
                inner = best.named_steps['clf']
                feature_order = list(getattr(inner, 'feature_names_in_', list(X.columns)))

            with open(features_path, 'w', encoding='utf8') as fh:
                json.dump(list(X.columns), fh)

            # Track selection
            results[name] = {
                'best_params': best_params,
                'metrics': metrics_mean,
                'model_path': str(model_path)
            }

            # choose best by ROC-AUC
            if metrics_mean['roc_auc_mean'] > best_score:
                best_score = metrics_mean['roc_auc_mean']
                best_overall = name

    # Summarize and save a small selection metadata file
    summary = {
        'best_model': best_overall,
        'best_score': best_score,
        'results': results
    }
    with open(model_dir / 'selection_summary.json', 'w', encoding='utf8') as fh:
        json.dump(summary, fh, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--cv', type=int, default=5)
    args = p.parse_args()
    res = train(Path(args.data), Path(args.model_dir), cv=args.cv)
    print('selection summary:', res)


if __name__ == '__main__':
    main()
