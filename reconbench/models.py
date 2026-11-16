"""Models under comparison and the metrics used everywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = ["Exact-match rules", "Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]


def make_lgbm(seed: int = 0) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9,
                          subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)


def get_models(seed: int = 0) -> dict:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1,
                                   class_weight="balanced_subsample", random_state=seed)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, tree_method="hist", eval_metric="logloss",
                                 random_state=seed),
        "LightGBM": make_lgbm(seed),
    }


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """Classic reconciliation rule: identical reference, or every key field identical."""
    keys = ((X["notional_exact"] == 1) & (X["trade_date_diff"] == 0) & (X["settle_date_diff"] == 0)
            & (X["cpty_token_set"] == 1) & (X["direction_match"] == 1) & (X["product_match"] == 1))
    return ((X["id_exact"] == 1) | keys).astype(float).to_numpy()


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5, no_id: np.ndarray | None = None) -> dict:
    pred = score >= threshold
    out = {"precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "pr_auc": average_precision_score(y, score)}
    if no_id is not None:  # recall restricted to true matches that have no usable identifier
        hard = (y == 1) & no_id
        out["recall_no_id"] = float(pred[hard].mean()) if hard.any() else float("nan")
    return out
