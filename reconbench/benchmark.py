"""The three benchmark protocols.

matched    train and test under the SAME corruption (the usual benchmark)
shift      train once on mild data (E2), test under every corruption
augmented  train once on the pooled data of all conditions, test under every corruption
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .generate import EXPERIMENTS, GenConfig, generate
from .models import get_models, make_lgbm, metrics, rule_baseline
from .pairs import build_features, candidate_pairs, pair_completeness, temporal_split

SEEDS = (0, 1, 2)
SHIFT_SOURCE = "E2 30% ids missing"


@lru_cache(maxsize=None)
def dataset(exp: str, seed: int) -> dict:
    a, b = generate(GenConfig(seed=seed, corruption=EXPERIMENTS[exp]))
    pairs = candidate_pairs(a, b)
    return {"pairs": pairs, "X": build_features(pairs), "y": pairs["is_match"].to_numpy(),
            "split": temporal_split(pairs), "completeness": pair_completeness(a, b, pairs),
            "n_a": len(a), "n_b": len(b)}


def _evaluate(model, d: dict) -> dict:
    te = d["split"]["test"]
    X, y = d["X"][te], d["y"][te]
    score = rule_baseline(X) if model is None else model.predict_proba(X)[:, 1]
    return metrics(y, score, no_id=(X["id_missing"] == 1).to_numpy())


def run_matched(log=print) -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            d = dataset(exp, seed)
            tr = d["split"]["train"]
            base = {"experiment": exp, "seed": seed, "pair_completeness": d["completeness"],
                    "candidate_pairs": len(d["y"]), "match_rate": float(d["y"].mean())}
            rows.append(base | {"model": "Exact-match rules"} | _evaluate(None, d))
            for name, model in get_models(seed).items():
                model.fit(d["X"][tr], d["y"][tr])
                rows.append(base | {"model": name} | _evaluate(model, d))
        log(f"  matched protocol: {exp} done")
    return pd.DataFrame(rows)


def run_shift_and_augmented(log=print) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        src = dataset(SHIFT_SOURCE, seed)
        shift_model = make_lgbm(seed).fit(src["X"][src["split"]["train"]], src["y"][src["split"]["train"]])
        Xs = pd.concat([dataset(e, seed)["X"][dataset(e, seed)["split"]["train"]] for e in EXPERIMENTS])
        ys = np.concatenate([dataset(e, seed)["y"][dataset(e, seed)["split"]["train"]] for e in EXPERIMENTS])
        aug_model = make_lgbm(seed).fit(Xs, ys)
        for exp in EXPERIMENTS:
            d = dataset(exp, seed)
            rows.append({"experiment": exp, "seed": seed, "protocol": "Trained on E2 only"} | _evaluate(shift_model, d))
            rows.append({"experiment": exp, "seed": seed, "protocol": "Trained on all conditions"} | _evaluate(aug_model, d))
        log(f"  shift / augmented protocols: seed {seed} done")
    return pd.DataFrame(rows)
