"""Real Time prediction and model usage

Usage:
    uv run python predict.py path/to/new_reviews.csv

"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

from src.features import MODELS_DIR, add_engineered_features


def main(csv_path: str) -> None:
    bundle = joblib.load(MODELS_DIR / "best_pipeline.joblib")
    pipeline = bundle["pipeline"]

    raw = pd.read_csv(csv_path)
    features = add_engineered_features(raw)
    preds = pipeline.predict(features)

    out = raw.copy()
    out["predicted_rating"] = preds.round(2)

    in_path = Path(csv_path)
    out_path = in_path.with_name(f"{in_path.stem}_scored{in_path.suffix}")
    out.to_csv(out_path, index=False)

    print(f"Scored {len(out)} rows using {bundle['model_name']} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
