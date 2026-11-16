"""Deep-dive analyses on one condition: SHAP, feature-group ablation, calibration,
cost-based thresholds and error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from .generate import FLAGS
from .models import make_lgbm, metrics
from .pairs import FEATURE_GROUPS, FEATURES, GROUP_OF

# Cost assumptions (in "analyst reviews"): see README for the reasoning.
COST_FP, COST_FN, COST_REVIEW = 50.0, 5.0, 1.0


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds) of the positive class, one row per pair."""
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def group_importance(values: np.ndarray) -> pd.Series:
    """Share of total mean |SHAP| carried by each feature group."""
    per_feature = pd.Series(np.abs(values).mean(0), index=FEATURES)
    per_group = per_feature.groupby(GROUP_OF).sum()
    return (per_group / per_group.sum()).reindex(FEATURE_GROUPS)


def ablation(d: dict, seed: int = 0) -> pd.DataFrame:
    """Retrain without each feature group, and with each group alone."""
    tr, te = d["split"]["train"], d["split"]["test"]
    y_tr, y_te = d["y"][tr], d["y"][te]

    def score(cols):
        m = make_lgbm(seed).fit(d["X"].loc[tr, cols], y_tr)
        return metrics(y_te, m.predict_proba(d["X"].loc[te, cols])[:, 1])

    full = score(FEATURES)
    rows = [{"group": "All features", "f1_without": full["f1"], "pr_auc_without": full["pr_auc"],
             "f1_alone": full["f1"], "pr_auc_alone": full["pr_auc"]}]
    for group, cols in FEATURE_GROUPS.items():
        without, alone = score([f for f in FEATURES if f not in cols]), score(cols)
        rows.append({"group": group, "f1_without": without["f1"], "pr_auc_without": without["pr_auc"],
                     "f1_alone": alone["f1"], "pr_auc_alone": alone["pr_auc"]})
    out = pd.DataFrame(rows)
    out["f1_drop"] = full["f1"] - out["f1_without"]
    return out


def calibrate(p_val: np.ndarray, y_val: np.ndarray, p_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Isotonic calibration fitted on validation; Brier score before / after on test."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
    return {"iso": iso, "brier_raw": brier_score_loss(y_test, p_test),
            "brier_calibrated": brier_score_loss(y_test, iso.predict(p_test))}


def reliability(p: np.ndarray, y: np.ndarray, bins=(0, .02, .1, .3, .5, .7, .9, .98, 1.0001)) -> pd.DataFrame:
    idx = np.digitize(p, bins) - 1
    rows = [{"mean_predicted": p[idx == i].mean(), "observed_rate": y[idx == i].mean(), "n": int((idx == i).sum())}
            for i in range(len(bins) - 1) if (idx == i).any()]
    return pd.DataFrame(rows)


def expected_cost(y: np.ndarray, p: np.ndarray, t: float, c_fp=COST_FP, c_fn=COST_FN) -> float:
    pred = p >= t
    return float(c_fp * (pred & (y == 0)).sum() + c_fn * (~pred & (y == 1)).sum())


def best_threshold(y: np.ndarray, p: np.ndarray, c_fp=COST_FP, c_fn=COST_FN) -> float:
    grid = np.linspace(0.01, 0.99, 99)
    return float(grid[int(np.argmin([expected_cost(y, p, t, c_fp, c_fn) for t in grid]))])


def three_zone_cost(y, p, lo, hi) -> dict:
    """Auto-match above `hi`, auto-reject below `lo`, send the band in between to an analyst
    (assumed to resolve it correctly at COST_REVIEW each)."""
    auto, reject, review = p >= hi, p < lo, (p >= lo) & (p < hi)
    fp, fn = int((auto & (y == 0)).sum()), int((reject & (y == 1)).sum())
    return {"lo": lo, "hi": hi, "cost": COST_FP * fp + COST_FN * fn + COST_REVIEW * int(review.sum()),
            "auto_matched": int(auto.sum()), "false_auto_matches": fp, "missed_matches": fn,
            "sent_to_review": int(review.sum()),
            "share_of_matches_automated": float((auto & (y == 1)).sum() / max((y == 1).sum(), 1)),
            "auto_match_precision": float((auto & (y == 1)).sum() / max(auto.sum(), 1))}


def best_three_zone(y, p) -> tuple[float, float]:
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    best = min(((lo, hi) for lo in grid for hi in grid if lo <= hi),
               key=lambda z: three_zone_cost(y, p, *z)["cost"])
    return best


def error_analysis(pairs: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Where do the mistakes come from?"""
    pred = p >= t
    pos = pairs[y == 1].copy()
    pos["found"] = pred[y == 1]
    flags_b = [f"{f}_b" for f in FLAGS]
    pos["n_corruptions"] = pos[flags_b].sum(axis=1)
    by_count = pos.groupby("n_corruptions")["found"].agg(matches="size", recall="mean")
    by_flag = pd.DataFrame({f.replace("_b", ""): {
        "matches_with": int(pos[f].sum()),
        "miss_rate_with": float(1 - pos.loc[pos[f] == 1, "found"].mean()) if pos[f].sum() else float("nan"),
        "miss_rate_without": float(1 - pos.loc[pos[f] == 0, "found"].mean()) if (pos[f] == 0).any() else float("nan"),
    } for f in flags_b}).T

    fp = (pred & (y == 0))
    lookalike = ((X["cpty_token_set"] >= 0.9) & (X["notional_rel_diff"] <= 0.005)
                 & (X["direction_match"] == 1) & (X["product_match"] == 1)).to_numpy()
    cols = ["record_id_a", "record_id_b", "trade_ref_a", "trade_ref_b", "counterparty_a", "counterparty_b",
            "notional_a", "notional_b", "rate_a", "rate_b", "trade_date_a", "trade_date_b"]
    sample = pd.concat([pairs[fp].assign(error="false positive", probability=p[fp]),
                        pairs[(~pred) & (y == 1)].assign(error="false negative", probability=p[(~pred) & (y == 1)])])
    return {"false_positives": int(fp.sum()), "false_negatives": int(((~pred) & (y == 1)).sum()),
            "fp_lookalike_share": float(lookalike[fp].mean()) if fp.any() else float("nan"),
            "lookalike_negatives": int((lookalike & (y == 0)).sum()),
            "by_corruption_count": by_count, "by_flag": by_flag,
            "sample": sample[["error", "probability"] + cols]}
