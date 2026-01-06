"""Data preparation for Heart Disease dataset.
Reads CSV, applies basic cleaning, encodes target as binary, and writes processed CSV.
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_process(input_path: Path) -> pd.DataFrame:
    # The UCI processed Cleveland dataset has no header; define columns per dataset docs
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(input_path, header=None, names=cols)

    # Replace missing values marked by '?' with NaN
    df.replace("?", pd.NA, inplace=True)
    # Convert numeric columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # The original target has values 0 (no disease) and 1-4 (disease levels). Convert to binary.
    df["target"] = (df["target"] > 0).astype(int)

    # Simple imputation: fill numeric NaNs with column median
    df = df.fillna(df.median(numeric_only=True))

    return df


def load_and_preprocess_data(data_path: str, test_size: float = 0.2):
    """Load processed data and split into train/test sets.

    Args:
        data_path: Path to the processed CSV file
        test_size: Proportion of data for testing

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]
    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, feature_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_process(Path(args.input))
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
