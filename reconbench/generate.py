"""Synthetic two-system trade data with CONTROLLABLE corruption.

Every kind of noise has its own knob in `Corruption`, so an experiment can turn
on exactly one problem (or all of them) and measure its effect. Each back-office
record also remembers which corruptions were applied to it (`c_*` columns), which
is what makes the error analysis possible.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

CCY_PAIRS = {
    "EUR/USD": 1.0850, "GBP/USD": 1.2700, "USD/JPY": 151.00, "EUR/GBP": 0.8550,
    "USD/CHF": 0.8800, "AUD/USD": 0.6600, "EUR/JPY": 164.00, "USD/CAD": 1.3600,
}
PRODUCTS = ["FX_SPOT", "FX_FORWARD", "FX_SWAP"]
COUNTERPARTIES = {
    "BNP PARIBAS": "BNPP", "BNP PARIBAS SECURITIES SERVICES": "BP2S",
    "SOCIETE GENERALE": "SOCGEN", "CREDIT AGRICOLE CIB": "CACIB", "NATIXIS": "NATX",
    "DEUTSCHE BANK": "DB", "COMMERZBANK": "COBA", "HSBC BANK": "HSBC",
    "HSBC CONTINENTAL EUROPE": "HSBCCE", "BARCLAYS BANK": "BARC",
    "BARCLAYS CAPITAL SECURITIES": "BARCAP", "JP MORGAN CHASE BANK": "JPM",
    "JP MORGAN SECURITIES": "JPMS", "GOLDMAN SACHS INTERNATIONAL": "GSI",
    "GOLDMAN SACHS BANK EUROPE": "GSBE", "MORGAN STANLEY": "MS", "CITIBANK": "CITI",
    "CITIGROUP GLOBAL MARKETS": "CGML", "BANK OF AMERICA": "BOFA", "UBS": "UBS",
    "NOMURA INTERNATIONAL": "NOMURA", "ING BANK": "ING", "SANTANDER": "SAN",
    "UNICREDIT": "UCG", "STANDARD CHARTERED BANK": "SCB", "ROYAL BANK OF CANADA": "RBC",
}
LEGAL_SUFFIXES = ["SA", "PLC", "AG", "LTD", "NV", "INC"]
FLAGS = ["c_id_missing", "c_id_typo", "c_amount", "c_date", "c_name", "c_rate_missing"]


@dataclass(frozen=True)
class Corruption:
    p_missing_id: float = 0.0
    p_id_typo: float = 0.0
    p_amount: float = 0.0          # share of records whose notional is off
    amount_scale: float = 0.005    # ... by up to this relative amount
    p_date: float = 0.0            # share of records whose trade date is shifted
    max_date_shift: int = 3        # ... by 1..max days
    p_name: float = 0.0            # share of records using a counterparty name variant
    p_rate_missing: float = 0.0


@dataclass(frozen=True)
class GenConfig:
    n_trades: int = 2500
    n_days: int = 30
    p_unconfirmed: float = 0.05    # in A, never reaches B
    p_orphan: float = 0.05         # in B, no trade in A
    seed: int = 0
    corruption: Corruption = Corruption()


# The benchmark conditions. E3-E5 add ONE problem on top of E2 (30% missing ids),
# because with every identifier present any other noise is irrelevant.
_E2 = Corruption(p_missing_id=0.30)
EXPERIMENTS = {
    "E0 clean": Corruption(),
    "E1 10% ids missing": Corruption(p_missing_id=0.10),
    "E2 30% ids missing": _E2,
    "E3 E2 + amount noise": replace(_E2, p_amount=0.20),
    "E4 E2 + name variants": replace(_E2, p_name=1.0),
    "E5 E2 + date shifts": replace(_E2, p_date=0.20),
    "E6 everything": Corruption(p_missing_id=0.50, p_id_typo=0.05, p_amount=0.20, p_date=0.20,
                                p_name=1.0, p_rate_missing=0.15),
}


def name_variant(name: str, rng) -> str:
    r = rng.random()
    if r < 0.30:
        return name.title()
    if r < 0.55:
        return f"{name} {rng.choice(LEGAL_SUFFIXES)}"
    if r < 0.75:
        return COUNTERPARTIES[name]            # short code: no characters in common
    if r < 0.88:
        return " ".join(name.split()[:2]).lower()  # truncated
    i = int(rng.integers(0, len(name) - 1))
    chars = list(name)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _make_trades(n: int, start: int, cfg: GenConfig, rng) -> pd.DataFrame:
    pairs, names = list(CCY_PAIRS), list(COUNTERPARTIES)
    pair_w = np.array([0.30, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08])
    cp_w = 1.0 / np.arange(1, len(names) + 1) ** 0.8
    cp_w /= cp_w.sum()
    ccy = rng.choice(pairs, n, p=pair_w)
    day = rng.integers(0, cfg.n_days, n)
    trade_date = np.busday_offset("2025-01-06", day).astype("datetime64[D]")
    product = rng.choice(PRODUCTS, n, p=[0.5, 0.35, 0.15])
    lag = rng.choice([30, 60, 90], n)
    settle = np.where(product == "FX_SPOT", np.busday_offset(trade_date, 2),
                      np.busday_offset(trade_date + lag.astype("timedelta64[D]"), 0, roll="forward"))
    sizes = np.array([0.5, 1, 2, 5, 10, 25]) * 1e6
    notional = np.where(rng.random(n) < 0.7, rng.choice(sizes, n, p=[0.15, 0.35, 0.2, 0.15, 0.1, 0.05]),
                        np.round(rng.lognormal(14.5, 1.0, n), -3))
    daily = {p: np.cumprod(1 + rng.normal(0, 0.003, cfg.n_days)) for p in pairs}
    rate = [round(CCY_PAIRS[p] * daily[p][d] * (1 + rng.normal(0, 0.00015)), 3 if "JPY" in p else 5)
            for p, d in zip(ccy, day)]
    return pd.DataFrame({
        "true_trade": np.arange(start, start + n),
        "trade_ref": [f"TRD-{x}" for x in rng.choice(np.arange(100000, 999999), n, replace=False)],
        "product": product, "ccy_pair": ccy, "direction": rng.choice(["BUY", "SELL"], n),
        "notional": notional.astype(float), "rate": rate,
        "trade_date": pd.to_datetime(trade_date), "settle_date": pd.to_datetime(settle.astype("datetime64[D]")),
        "counterparty": rng.choice(names, n, p=cp_w),
    })


def _corrupt(row: dict, c: Corruption, rng) -> dict:
    r = dict(row) | {f: 0 for f in FLAGS}
    digits = r["trade_ref"].split("-")[1]
    if rng.random() < c.p_missing_id:
        r["trade_ref"], r["c_id_missing"] = None, 1
    else:
        if rng.random() < c.p_id_typo:
            i = int(rng.integers(0, len(digits)))
            digits = digits[:i] + str((int(digits[i]) + int(rng.integers(1, 9))) % 10) + digits[i + 1:]
            r["c_id_typo"] = 1
        r["trade_ref"] = rng.choice([f"TRD-{digits}", digits, f"REF/{digits}"], p=[0.5, 0.3, 0.2])
    if rng.random() < c.p_amount:
        r["notional"] = round(r["notional"] * (1 + rng.uniform(-c.amount_scale, c.amount_scale)), 2)
        r["c_amount"] = 1
    if rng.random() < c.p_date:
        shift = int(rng.integers(1, c.max_date_shift + 1)) * int(rng.choice([-1, 1]))
        r["trade_date"] = r["trade_date"] + pd.Timedelta(days=shift)
        r["c_date"] = 1
    if rng.random() < c.p_name:
        r["counterparty"] = name_variant(r["counterparty"], rng)
        r["c_name"] = 1
    r["rate"] = round(r["rate"], 2 if "JPY" in r["ccy_pair"] else 4)  # B always stores fewer decimals
    if rng.random() < c.p_rate_missing:
        r["rate"], r["c_rate_missing"] = np.nan, 1
    return r


def generate(cfg: GenConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (system_a, system_b). The same seed gives the same underlying trades
    whatever the corruption, so conditions are comparable trade for trade."""
    trade_rng = np.random.default_rng(cfg.seed)
    trades = _make_trades(cfg.n_trades, 0, cfg, trade_rng)
    orphans = _make_trades(int(cfg.n_trades * cfg.p_orphan), cfg.n_trades, cfg, trade_rng)
    confirmed = trades[trade_rng.random(len(trades)) >= cfg.p_unconfirmed]

    a = trades.copy()
    a.insert(0, "record_id", [f"A-{i:06d}" for i in range(len(a))])
    noise_rng = np.random.default_rng(cfg.seed + 10_000)
    b = pd.DataFrame([_corrupt(row, cfg.corruption, noise_rng)
                      for row in pd.concat([confirmed, orphans]).to_dict("records")])
    b = b.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    b.insert(0, "record_id", [f"B-{i:06d}" for i in range(len(b))])
    return a, b
