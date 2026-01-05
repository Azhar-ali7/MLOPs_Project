"""Data preparation for Heart Disease dataset.
Reads CSV, applies basic cleaning, encodes target as binary, and writes processed CSV.
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def load_and_process(input_path: Path) -> pd.DataFrame:
    # The UCI processed Cleveland dataset has no header; define columns per dataset docs
    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(input_path, header=None, names=cols)

    # Replace missing values marked by '?' with NaN
    df.replace('?', pd.NA, inplace=True)
    # Convert numeric columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # The original target has values 0 (no disease) and 1-4 (disease levels). Convert to binary.
    df['target'] = (df['target'] > 0).astype(int)

    # Simple imputation: fill numeric NaNs with column median
    df = df.fillna(df.median(numeric_only=True))

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_process(Path(args.input))
    df.to_csv(out, index=False)


if __name__ == '__main__':
    main()
