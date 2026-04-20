from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


NUM_COLS = [
    "num_helpful_votes", "user_age",
    "review_length", "review_word_count",
    "review_year", "review_month", "review_dayofweek",
    "verified_purchase", "app_major_version",
]
# low and high cardinality of features: app_category just 18->low but review_text->739
LOW_CARD_COLS = ["app_category", "review_language", "device_type", "user_gender"]
HIGH_CARD_COLS = ["app_name", "user_country", "review_text"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive text/date/version features; drop identifier and replaced columns.
    """
    
    d = df.copy()
    d["review_length"] = d["review_text"].fillna("").str.len()
    d["review_word_count"] = d["review_text"].fillna("").str.split().str.len().fillna(0)

    dt = pd.to_datetime(d["review_date"])
    d["review_year"] = dt.dt.year
    d["review_month"] = dt.dt.month
    d["review_dayofweek"] = dt.dt.dayofweek

    d["app_major_version"] = (
        d["app_version"].fillna("0").str.extract(r"(\d+)")
          .astype(float).fillna(0).astype(int)
    )

    d["verified_purchase"] = d["verified_purchase"].astype(int)

    return d.drop(columns=["review_id", "user_id", "review_date", "app_version"])


def make_preprocessor() -> ColumnTransformer:
    """Return a fresh ColumnTransformer. One instance per CV fold, no leakage."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    low_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num",      numeric,   NUM_COLS),
        ("low_cat",  low_card,  LOW_CARD_COLS),
        ("high_cat", high_card, HIGH_CARD_COLS),
    ])
