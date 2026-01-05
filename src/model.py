"""Model loader and predictor utilities."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


class ModelWrapper:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None

    def load(self):
        if not self._model:
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, df: pd.DataFrame) -> list:
        model = self.load()

        # If the fitted estimator exposes feature_names_in_, use it to validate/reorder inputs
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            expected = list(expected)
            provided = list(df.columns)
            if set(expected) != set(provided):
                raise ValueError("Feature names mismatch. " f"Expected: {expected}. Got: {provided}.")
            # Reorder columns to match training order
            if provided != expected:
                df = df[expected]

        # Best-effort: ensure numeric dtype where appropriate
        try:
            df = df.astype(float)
        except Exception:
            # Let the model raise a clear error if types are incompatible
            pass

        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        return [{"prediction": int(p), "probability": float(prob)} for p, prob in zip(preds, probs)]
