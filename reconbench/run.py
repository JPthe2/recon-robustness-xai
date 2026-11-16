"""Run every experiment and write results/:  python -m reconbench.run"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from . import analysis as an
from . import viz
from .benchmark import SEEDS, dataset, run_matched, run_shift_and_augmented
from .generate import EXPERIMENTS
from .models import MODEL_NAMES, make_lgbm, metrics
from .pairs import FEATURE_GROUPS, FEATURES

plt = viz.plt
RESULTS = Path(__file__).resolve().parents[1] / "results"
SHORT = {e: e.split()[0] for e in EXPERIMENTS}
DEEP_DIVE = "E6 everything"


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"seeds": list(SEEDS), "experiments": {e: vars(c) for e, c in EXPERIMENTS.items()}}

    print("1/4 matched protocol")
    matched = run_matched()
    matched.to_csv(RESULTS / "matched_runs.csv", index=False)
    summary = matched.groupby(["experiment", "model"])[["precision", "recall", "f1", "pr_auc", "recall_no_id"]] \
                     .agg(["mean", "std"])
    out["matched"] = {f"{e} | {m}": {f"{a}_{b}": float(v) for (a, b), v in row.items()}
                      for (e, m), row in summary.iterrows()}
    out["blocking"] = matched.groupby("experiment")[["pair_completeness", "candidate_pairs", "match_rate"]] \
                             .mean().to_dict("index")
    _fig_robustness(matched)
    _fig_no_id(matched)

    print("2/4 shift and augmented protocols")
    shift = run_shift_and_augmented()
    shift.to_csv(RESULTS / "shift_runs.csv", index=False)
    lg = matched[matched["model"] == "LightGBM"].assign(protocol="Trained on the same condition")
    protocols = pd.concat([lg, shift])
    out["protocols"] = {f"{e} | {p}": {"f1": float(g["f1"].mean()), "recall": float(g["recall"].mean()),
                                       "precision": float(g["precision"].mean())}
                        for (e, p), g in protocols.groupby(["experiment", "protocol"])}
    _fig_protocols(protocols)

    print("3/4 SHAP across conditions")
    shares = {}
    for exp in EXPERIMENTS:
        d = dataset(exp, 0)
        tr, te = d["split"]["train"], d["split"]["test"]
        model = make_lgbm(0).fit(d["X"][tr], d["y"][tr])
        sample = d["X"][te].sample(min(20000, int(te.sum())), random_state=0)
        shares[exp] = an.group_importance(an.shap_values(model, sample))
    shares = pd.DataFrame(shares).T
    out["shap_group_share"] = shares.to_dict("index")
    _fig_shap_groups(shares)

    print("4/4 deep dive on", DEEP_DIVE)
    d = dataset(DEEP_DIVE, 0)
    fit, val, te = d["split"]["fit"], d["split"]["val"], d["split"]["test"]
    X, y, pairs = d["X"], d["y"], d["pairs"]
    model = make_lgbm(0).fit(X[fit], y[fit])
    p_val, p_te = model.predict_proba(X[val])[:, 1], model.predict_proba(X[te])[:, 1]

    sv = an.shap_values(model, X[te])
    out["shap_feature_importance"] = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False).to_dict()
    plt.figure()
    shap.summary_plot(sv, X[te], show=False, max_display=13, plot_size=(8, 5.2))
    plt.title("SHAP summary: each dot is one test pair", loc="left", fontweight="bold")
    plt.savefig(RESULTS / "shap_beeswarm.png", bbox_inches="tight", facecolor=viz.SURFACE)
    plt.close("all")

    abl = an.ablation(d)
    out["ablation"] = abl.set_index("group").to_dict("index")
    _fig_ablation(abl)

    cal = an.calibrate(p_val, y[val], p_te, y[te])
    out["calibration"] = {"brier_raw": cal["brier_raw"], "brier_calibrated": cal["brier_calibrated"]}
    _fig_calibration(an.reliability(p_te, y[te]), an.reliability(cal["iso"].predict(p_te), y[te]), out["calibration"])

    # Thresholds are chosen on VALIDATION and judged on TEST. Errors are rare, so the
    # predictions of the three seeds are pooled to get stable counts.
    yv, pv, yt, pt = [y[val]], [p_val], [y[te]], [p_te]
    pairs_te, X_te = [pairs[te]], [X[te]]
    for seed in SEEDS[1:]:
        ds = dataset(DEEP_DIVE, seed)
        ms = make_lgbm(seed).fit(ds["X"][ds["split"]["fit"]], ds["y"][ds["split"]["fit"]])
        yv.append(ds["y"][ds["split"]["val"]]); pv.append(ms.predict_proba(ds["X"][ds["split"]["val"]])[:, 1])
        yt.append(ds["y"][ds["split"]["test"]]); pt.append(ms.predict_proba(ds["X"][ds["split"]["test"]])[:, 1])
        pairs_te.append(ds["pairs"][ds["split"]["test"]]); X_te.append(ds["X"][ds["split"]["test"]])
    yv, pv, yt, pt = map(np.concatenate, (yv, pv, yt, pt))
    grid = np.linspace(0.01, 0.99, 99)
    t_cost = an.best_threshold(yv, pv)
    t_f1 = float(grid[int(np.argmax([metrics(yv, pv, t)["f1"] for t in grid]))])
    policies = {}
    for name, t in [("Default threshold", 0.5), ("Best F1 on validation", t_f1),
                    ("Lowest cost on validation", t_cost)]:
        m = metrics(yt, pt, t)
        policies[name] = m | {"threshold": t, "cost": an.expected_cost(yt, pt, t),
                              "false_positives": int(((pt >= t) & (yt == 0)).sum()),
                              "false_negatives": int(((pt < t) & (yt == 1)).sum())}
    lo, hi = an.best_three_zone(yv, pv)
    zone = an.three_zone_cost(yt, pt, lo, hi)
    sens = {str(r): an.best_threshold(yv, pv, c_fp=r, c_fn=1.0) for r in [0.2, 1, 5, 10, 50, 200]}
    confident_fp = int(((pt >= 0.99) & (yt == 0)).sum())
    out["thresholds"] = {"costs": {"false_positive": an.COST_FP, "false_negative": an.COST_FN, "review": an.COST_REVIEW},
                         "single_threshold": policies, "three_zone": zone, "threshold_vs_cost_ratio": sens,
                         "test_pairs": int(len(yt)), "test_matches": int(yt.sum()),
                         "validation_matches": int(yv.sum()),
                         "false_positives_scored_above_0.99": confident_fp}
    _fig_cost(yt, pt, policies)

    err = an.error_analysis(pd.concat(pairs_te, ignore_index=True), pd.concat(X_te, ignore_index=True), yt, pt, t_cost)
    err["sample"].sort_values("probability").to_csv(RESULTS / "errors.csv", index=False)
    out["errors"] = {"threshold": t_cost, "false_positives": err["false_positives"],
                     "false_negatives": err["false_negatives"], "fp_lookalike_share": err["fp_lookalike_share"],
                     "lookalike_negatives": err["lookalike_negatives"],
                     "recall_by_corruption_count": err["by_corruption_count"].to_dict("index"),
                     "miss_rate_by_corruption": err["by_flag"].to_dict("index")}
    _fig_errors(err)
    _fig_local(model, X[te], y[te], p_te, sv, t_cost)

    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    _write_tables(matched, protocols, out)
    print("done ->", RESULTS)


# ---------------------------------------------------------------- tables
def _write_tables(matched, protocols, out):
    f1 = matched.groupby(["experiment", "model"])["f1"].agg(["mean", "std"])
    cell = (f1["mean"].map("{:.3f}".format) + " ± " + f1["std"].map("{:.3f}".format)).unstack()[MODEL_NAMES]
    noid = matched.groupby(["experiment", "model"])["recall_no_id"].mean().unstack()[MODEL_NAMES].round(3)
    prot = protocols.groupby(["experiment", "protocol"])["f1"].mean().unstack().round(3)
    text = ["## F1 by condition (mean ± std over seeds)", cell.to_markdown(),
            "## Recall on true matches without identifier", noid.to_markdown(),
            "## LightGBM F1 by training protocol", prot.to_markdown(),
            "## Ablation (E6)", pd.DataFrame(out["ablation"]).T.round(4).to_markdown(),
            "## SHAP share by group", pd.DataFrame(out["shap_group_share"]).T.round(3).to_markdown(),
            "## Threshold policies (E6 test)", pd.DataFrame(out["thresholds"]["single_threshold"]).T.round(4).to_markdown()]
    (RESULTS / "tables.md").write_text("\n\n".join(text), encoding="utf-8")


# ---------------------------------------------------------------- figures
def _fig_robustness(matched):
    agg = matched.groupby(["experiment", "model"])["f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(EXPERIMENTS))
    for name in MODEL_NAMES:
        g = agg[agg["model"] == name].set_index("experiment").loc[list(EXPERIMENTS)]
        ax.errorbar(x, g["mean"], yerr=g["std"], marker="o", markersize=6, capsize=3,
                    color=viz.MODEL_COLORS[name], label=name)
    ax.set_xticks(x, [e.replace(" ", "\n", 1) for e in EXPERIMENTS], fontsize=8.5)
    ax.set_ylabel("F1 on the test set (mean of 3 seeds, bars = std)")
    ax.set_title("How each model degrades as data quality gets worse")
    ax.legend(loc="lower left", ncol=2)
    viz.save(fig, RESULTS / "robustness_f1.png")


def _fig_no_id(matched):
    agg = matched.groupby(["experiment", "model"])["recall_no_id"].mean().unstack()
    exps = [e for e in EXPERIMENTS if not e.startswith("E0")]
    fig, ax = plt.subplots(figsize=(9, 4))
    x, w = np.arange(len(exps)), 0.26
    for i, name in enumerate(["Exact-match rules", "Logistic Regression", "LightGBM"]):
        vals = agg.loc[exps, name]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=viz.MODEL_COLORS[name], label=name)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [e.replace(" ", "\n", 1) for e in exps], fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("recall")
    ax.set_title("Recall on the hard cases only: true matches with no identifier")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    viz.save(fig, RESULTS / "recall_no_id.png")


def _fig_protocols(protocols):
    agg = protocols.groupby(["experiment", "protocol"])["f1"].mean().unstack().loc[list(EXPERIMENTS)]
    order = ["Trained on E2 only", "Trained on the same condition", "Trained on all conditions"]
    colors = [viz.ORANGE, viz.BLUE, viz.AQUA]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x, w = np.arange(len(agg)), 0.27
    for i, (p, c) in enumerate(zip(order, colors)):
        ax.bar(x + (i - 1) * w, agg[p], width=w - 0.02, color=c, label=p)
        for xi, v in zip(x, agg[p]):
            ax.text(xi + (i - 1) * w, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [e.replace(" ", "\n", 1) for e in agg.index], fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("LightGBM F1 on the test set")
    ax.set_title("Does a model trained on mild data survive dirtier data?")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    viz.save(fig, RESULTS / "train_test_shift.png")


def _fig_shap_groups(shares):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = np.zeros(len(shares))
    labels = [e for e in shares.index][::-1]
    data = shares.loc[labels]
    for group in FEATURE_GROUPS:
        ax.barh(labels, data[group], left=left, color=viz.GROUP_COLORS[group], label=group,
                height=0.62, edgecolor=viz.SURFACE, linewidth=1.5)
        for i, (v, l) in enumerate(zip(data[group], left)):
            if v >= 0.07:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if group in ("Identifier", "Direction/product") else viz.INK)
        left += data[group].to_numpy()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib_pct())
    ax.set_xlabel("share of total mean |SHAP|")
    ax.set_title("What the model relies on, condition by condition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, fontsize=8.5)
    ax.grid(visible=False)
    viz.save(fig, RESULTS / "shap_groups.png")


def matplotlib_pct():
    from matplotlib.ticker import PercentFormatter
    return PercentFormatter(1.0)


def _fig_ablation(abl):
    a = abl[abl["group"] != "All features"].sort_values("f1_drop")
    full = abl.loc[abl["group"] == "All features", "f1_alone"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    axes[0].barh(a["group"], a["f1_drop"], color=viz.BLUE, height=0.6)
    for i, v in enumerate(a["f1_drop"]):
        axes[0].text(max(v, 0) + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(min(a["f1_drop"].min(), 0) - 0.005, a["f1_drop"].max() * 1.3 + 0.01)
    axes[0].set_title("F1 lost when the group is REMOVED")
    axes[1].barh(a["group"], a["f1_alone"], color=viz.ORANGE, height=0.6)
    for i, v in enumerate(a["f1_alone"]):
        axes[1].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].axvline(full, color=viz.MUTED, ls="--", lw=1.2)
    axes[1].text(full, len(a) - 0.4, f"all features {full:.3f}", ha="right", fontsize=8.5, color=viz.MUTED)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("F1 with ONLY this group")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    viz.save(fig, RESULTS / "ablation.png")


def _fig_calibration(raw, cal, brier):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=viz.MUTED, ls="--", lw=1.2, label="Perfect calibration")
    ax.plot(raw["mean_predicted"], raw["observed_rate"], marker="o", color=viz.BLUE,
            label=f"LightGBM raw (Brier {brier['brier_raw']:.5f})")
    ax.plot(cal["mean_predicted"], cal["observed_rate"], marker="o", color=viz.ORANGE,
            label=f"After isotonic (Brier {brier['brier_calibrated']:.5f})")
    ax.set_xlabel("predicted match probability")
    ax.set_ylabel("observed share of real matches")
    ax.set_title("Can the probabilities be trusted?")
    ax.legend(loc="upper left", fontsize=8.5)
    viz.save(fig, RESULTS / "calibration.png")


def _fig_cost(y, p, policies):
    grid = np.linspace(0.01, 0.99, 99)
    cost = [an.expected_cost(y, p, t) for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid, cost, color=viz.BLUE)
    for (name, pol), c in zip(policies.items(), [viz.MUTED, viz.AQUA, viz.ORANGE]):
        ax.scatter([pol["threshold"]], [pol["cost"]], s=70, color=c, zorder=3,
                   label=f"{name} ({pol['threshold']:.2f}): cost {pol['cost']:.0f}")
    ax.set_ylim(0, max(cost) * 1.1)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("total cost on the pooled test sets")
    ax.set_title(f"Cost of errors when a false match costs {an.COST_FP:.0f} and a miss costs {an.COST_FN:.0f}")
    ax.legend(loc="lower center", fontsize=8.5)
    viz.save(fig, RESULTS / "cost_curve.png")


def _fig_errors(err):
    bc = err["by_corruption_count"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    miss = 1 - bc["recall"]
    axes[0].bar(bc.index.astype(str), miss, color=viz.BLUE, width=0.6)
    for i, (v, n) in enumerate(zip(miss, bc["matches"])):
        axes[0].text(i, v + miss.max() * 0.03, f"{v:.1%}\n(n={n})", ha="center", fontsize=8.5)
    axes[0].set_ylim(0, miss.max() * 1.35 + 0.001)
    axes[0].set_xlabel("number of corruptions on the back-office record")
    axes[0].set_ylabel("share of true matches missed")
    axes[0].set_title("Misses only appear from 3 stacked corruptions")
    axes[0].grid(axis="x", visible=False)
    bf = err["by_flag"].dropna()
    labels = {"c_id_missing": "ID missing", "c_id_typo": "ID typo", "c_amount": "Amount noise",
              "c_date": "Date shift", "c_name": "Name variant", "c_rate_missing": "Rate missing"}
    yy = np.arange(len(bf))
    axes[1].barh(yy + 0.2, bf["miss_rate_with"], height=0.36, color=viz.ORANGE, label="records WITH this corruption")
    axes[1].barh(yy - 0.2, bf["miss_rate_without"], height=0.36, color=viz.MUTED, label="records without it")
    axes[1].set_yticks(yy, [labels[i] for i in bf.index])
    axes[1].set_xlabel("share of true matches missed")
    axes[1].set_title("Which corruption causes misses")
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(axis="y", visible=False)
    viz.save(fig, RESULTS / "error_analysis.png")


def _fig_local(model, X, y, p, sv, t):
    """Three individual explanations: a match found without identifier, a false positive, a false negative."""
    Xr = X.reset_index(drop=True)
    cases = {"Correct match, no identifier": np.where((y == 1) & (p >= 0.9) & (Xr["id_missing"] == 1))[0],
             "False positive": np.where((y == 0) & (p >= t))[0],
             "False negative": np.where((y == 1) & (p < t))[0]}
    cases = {k: v for k, v in cases.items() if len(v)}
    fig, axes = plt.subplots(1, len(cases), figsize=(5.3 * len(cases), 4.6))
    for ax, (name, idx) in zip(np.atleast_1d(axes), cases.items()):
        i = idx[np.argmax(p[idx])] if name == "False positive" else idx[np.argmin(p[idx])] if name == "False negative" else idx[0]
        s = pd.Series(sv[i], index=FEATURES)
        s = s[s.abs().sort_values().index][-8:]
        labels = [f"{f} = {Xr.loc[i, f]:.4g}" for f in s.index]
        ax.barh(labels, s.values, color=[viz.BLUE if v >= 0 else viz.ORANGE for v in s.values], height=0.6)
        ax.axvline(0, color=viz.MUTED, lw=1)
        ax.set_title(f"{name}\npredicted probability {p[i]:.2f}", fontsize=10.5)
        ax.set_xlabel("SHAP value: blue = towards match, orange = against")
        ax.grid(axis="y", visible=False)
    viz.save(fig, RESULTS / "shap_local.png")


if __name__ == "__main__":
    main()
