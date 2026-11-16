from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reconbench import analysis as an
from reconbench.generate import EXPERIMENTS, FLAGS, Corruption, GenConfig, generate
from reconbench.models import make_lgbm, metrics, rule_baseline
from reconbench.pairs import (FEATURE_GROUPS, FEATURES, build_features, candidate_pairs,
                              pair_completeness, temporal_split)

SMALL = GenConfig(n_trades=500, n_days=20, seed=4)


def prep(corruption):
    a, b = generate(replace(SMALL, corruption=corruption))
    pairs = candidate_pairs(a, b)
    return a, b, pairs, build_features(pairs), pairs["is_match"].to_numpy(), temporal_split(pairs)


@pytest.fixture(scope="module")
def dirty():
    return prep(EXPERIMENTS["E6 everything"])


def test_clean_condition_has_no_corruption_and_same_trades_as_dirty(dirty):
    a0, b0 = generate(SMALL)
    assert b0[FLAGS].to_numpy().sum() == 0 and b0["trade_ref"].notna().all()
    pd.testing.assert_frame_equal(a0, dirty[0])  # same underlying trades whatever the corruption


def test_each_knob_only_touches_its_own_field():
    _, b = generate(replace(SMALL, corruption=Corruption(p_amount=1.0)))
    assert b["c_amount"].all() and b[[f for f in FLAGS if f != "c_amount"]].to_numpy().sum() == 0
    _, b = generate(replace(SMALL, corruption=Corruption(p_missing_id=1.0)))
    assert b["trade_ref"].isna().all()


def test_blocking_survives_three_day_date_shifts(dirty):
    a, b, pairs, *_ = dirty
    assert pair_completeness(a, b, pairs) == 1.0


def test_split_is_leak_free_and_identical_across_conditions(dirty):
    _, _, pairs, _, _, split = dirty
    fit, val, test = split["fit"], split["val"], split["test"]
    assert not (fit & val).any() and not (val & test).any() and not (split["train"] & test).any()
    for side in ("a", "b"):
        col = f"true_trade_{side}"
        assert not set(pairs.loc[split["train"], col]) & set(pairs.loc[test, col])
    _, _, clean_pairs, _, _, clean_split = prep(Corruption())
    assert set(pairs.loc[test, "true_trade_a"]) == set(clean_pairs.loc[clean_split["test"], "true_trade_a"])


def test_feature_groups_cover_every_feature_once():
    grouped = [f for fs in FEATURE_GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(FEATURES) and len(set(grouped)) == len(grouped)


def test_model_beats_rules_on_dirty_data_and_shap_is_additive(dirty):
    _, _, _, X, y, split = dirty
    model = make_lgbm().fit(X[split["train"]], y[split["train"]])
    Xt, yt = X[split["test"]], y[split["test"]]
    p = model.predict_proba(Xt)[:, 1]
    assert metrics(yt, p)["f1"] > metrics(yt, rule_baseline(Xt))["f1"]
    sv = an.shap_values(model, Xt)
    assert sv.shape == Xt.shape
    # SHAP values + base value must add up to the model's raw (log-odds) output
    raw = model.predict(Xt, raw_score=True)
    assert np.allclose(sv.sum(1) + (raw - sv.sum(1)).mean(), raw, atol=1e-4)
    shares = an.group_importance(sv)
    assert shares.sum() == pytest.approx(1.0) and list(shares.index) == list(FEATURE_GROUPS)


def test_cost_functions_on_a_known_case():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.95, 0.40, 0.97, 0.30, 0.02])
    assert an.expected_cost(y, p, 0.5, c_fp=50, c_fn=5) == 55      # one false match + one miss
    zone = an.three_zone_cost(y, p, lo=0.1, hi=0.99)
    assert zone["sent_to_review"] == 4 and zone["false_auto_matches"] == 0 and zone["cost"] == 4
    assert an.best_threshold(y, p, c_fp=1000, c_fn=1) > 0.97         # expensive false matches -> strict
    assert an.best_threshold(y, p, c_fp=0.001, c_fn=1) <= 0.4        # cheap false matches -> keep the 0.40 match
