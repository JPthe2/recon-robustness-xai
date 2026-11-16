"""Blocking, pairwise features and the leak-free temporal split."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .generate import FLAGS

LEGAL_TOKENS = {"SA", "PLC", "AG", "LTD", "NV", "INC", "LLC", "SE"}
BLOCK_WINDOW = 3  # days

FEATURE_GROUPS = {
    "Identifier": ["id_exact", "id_similarity", "id_missing"],
    "Amount": ["notional_rel_diff", "notional_exact"],
    "Dates": ["trade_date_diff", "settle_date_diff"],
    "Rate": ["rate_rel_diff", "rate_missing"],
    "Counterparty": ["cpty_jaro_winkler", "cpty_token_set"],
    "Direction/product": ["direction_match", "product_match"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
A_COLS = ["record_id", "true_trade", "trade_ref", "product", "ccy_pair", "direction", "notional",
          "rate", "trade_date", "settle_date", "counterparty"]


def norm_id(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return re.sub(r"\D", "", str(x)) or None


def norm_name(x) -> str:
    tokens = re.sub(r"[^A-Z0-9 ]", " ", str(x).upper()).split()
    return " ".join([t for t in tokens if t not in LEGAL_TOKENS] or tokens)


def candidate_pairs(a: pd.DataFrame, b: pd.DataFrame, window: int = BLOCK_WINDOW) -> pd.DataFrame:
    """Same currency pair and trade dates within `window` days, OR same normalised reference."""
    ka, kb = a[["record_id", "ccy_pair", "trade_date"]], b[["record_id", "ccy_pair", "trade_date"]]
    p1 = ka.merge(kb, on="ccy_pair", suffixes=("_a", "_b"))
    p1 = p1.loc[(p1["trade_date_a"] - p1["trade_date_b"]).abs().dt.days <= window, ["record_id_a", "record_id_b"]]
    ia = a.assign(_id=a["trade_ref"].map(norm_id)).dropna(subset=["_id"])[["record_id", "_id"]]
    ib = b.assign(_id=b["trade_ref"].map(norm_id)).dropna(subset=["_id"])[["record_id", "_id"]]
    p2 = ia.merge(ib, on="_id", suffixes=("_a", "_b"))[["record_id_a", "record_id_b"]]
    keys = pd.concat([p1, p2]).drop_duplicates()
    pairs = (keys.merge(a[A_COLS].add_suffix("_a"), on="record_id_a")
                 .merge(b[A_COLS + FLAGS].add_suffix("_b"), on="record_id_b"))
    pairs["is_match"] = (pairs["true_trade_a"] == pairs["true_trade_b"]).astype(int)
    return pairs.reset_index(drop=True)


def pair_completeness(a: pd.DataFrame, b: pd.DataFrame, pairs: pd.DataFrame) -> float:
    """Share of the true matches that survive blocking."""
    return float(pairs["is_match"].sum() / b["true_trade"].isin(a["true_trade"]).sum())


def _cached(func, left, right) -> np.ndarray:
    cache = {k: func(*k) for k in set(zip(left, right))}
    return np.array([cache[k] for k in zip(left, right)])


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=pairs.index)
    ida, idb = pairs["trade_ref_a"].map(norm_id), pairs["trade_ref_b"].map(norm_id)
    missing = ida.isna() | idb.isna()
    f["id_exact"] = ((ida == idb) & ~missing).astype(int)
    f["id_similarity"] = [0.0 if m else fuzz.ratio(x, y) / 100 for x, y, m in zip(ida, idb, missing)]
    f["id_missing"] = missing.astype(int)
    na, nb = pairs["notional_a"].astype(float), pairs["notional_b"].astype(float)
    f["notional_rel_diff"] = (na - nb).abs() / np.maximum(na, nb)
    f["notional_exact"] = (na == nb).astype(int)
    f["trade_date_diff"] = (pairs["trade_date_a"] - pairs["trade_date_b"]).abs().dt.days
    f["settle_date_diff"] = (pairs["settle_date_a"] - pairs["settle_date_b"]).abs().dt.days
    ra, rb = pairs["rate_a"].astype(float), pairs["rate_b"].astype(float)
    f["rate_rel_diff"] = (ra - rb).abs() / ra
    f["rate_missing"] = f["rate_rel_diff"].isna().astype(int)
    ca, cb = pairs["counterparty_a"].map(norm_name).tolist(), pairs["counterparty_b"].map(norm_name).tolist()
    f["cpty_jaro_winkler"] = _cached(JaroWinkler.normalized_similarity, ca, cb)
    f["cpty_token_set"] = _cached(fuzz.token_set_ratio, ca, cb) / 100
    f["direction_match"] = (pairs["direction_a"] == pairs["direction_b"]).astype(int)
    f["product_match"] = (pairs["product_a"] == pairs["product_b"]).astype(int)
    return f[FEATURES].astype(float)


def temporal_split(pairs: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.2) -> dict:
    """Masks for fit / validation / test, by SYSTEM A trade date.

    System A dates are clean, so the same trade always lands in the same split
    whatever corruption system B received. A pair is used only if both of its
    underlying trades are in the same split, hence no trade is seen on two sides.
    `train` = fit + val (used when no validation set is needed).
    """
    days = np.sort(pairs["trade_date_a"].unique())
    test_start = days[int(len(days) * train_frac)]
    val_start = days[int(len(days) * train_frac * (1 - val_frac))]
    clean_date = pairs.groupby("true_trade_a")["trade_date_a"].first()
    date_b = pairs["true_trade_b"].map(clean_date).fillna(pairs["trade_date_b"])  # orphans keep their own date

    def part(d):
        return np.where(d < val_start, 0, np.where(d < test_start, 1, 2))

    pa, pb = part(pairs["trade_date_a"]), part(date_b)
    same = pa == pb
    return {"fit": same & (pa == 0), "val": same & (pa == 1), "test": same & (pa == 2),
            "train": (pa < 2) & (pb < 2)}
