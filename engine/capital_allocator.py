"""
Capital allocation and position sizing.
Implements the 'capital-allocator' skill from jackson-video-resources/skills:
  - Kelly Criterion (full and fractional)
  - Fixed fractional (existing, extended here)
  - Volatility targeting
  - Multi-strategy allocation
  - Mean-variance optimization (Markowitz)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: f* = (bp - q) / b
      b = avg_win / avg_loss (odds ratio)
      p = win_rate
      q = 1 - win_rate
    Returns fraction of capital to risk. Capped at 0.25 (quarter-Kelly safety).
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    f = (b * p - q) / b
    return float(np.clip(f, 0, 0.25))  # quarter-Kelly cap


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return kelly_fraction(win_rate, avg_win, avg_loss) * 0.5


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------

def vol_target_size(
    capital: float,
    target_vol: float,
    realized_vol: float,
    price: float,
) -> int:
    """
    Scale position so portfolio vol ≈ target_vol.
    shares = (capital * target_vol / realized_vol) / price
    """
    if realized_vol <= 0 or price <= 0:
        return 1
    notional = capital * (target_vol / realized_vol)
    return max(1, int(notional / price))


# ---------------------------------------------------------------------------
# Multi-strategy allocation (equal risk / Sharpe-weighted)
# ---------------------------------------------------------------------------

def equal_risk_weights(sharpes: list[float]) -> list[float]:
    """
    Weight each strategy proportional to its Sharpe ratio.
    Strategies with Sharpe ≤ 0 receive zero allocation.
    """
    s = np.array([max(0, x) for x in sharpes], dtype=float)
    total = s.sum()
    if total == 0:
        n = len(sharpes)
        return [1 / n] * n
    return (s / total).tolist()


def allocation_report(
    strategies: list[dict],
    total_capital: float,
) -> pd.DataFrame:
    """
    Given a list of strategy dicts with keys: name, sharpe, max_dd, win_rate, avg_win, avg_loss
    Returns allocation DataFrame with: weight, allocated_capital, kelly_f, recommendation
    """
    sharpes = [s.get("sharpe", 0) for s in strategies]
    weights = equal_risk_weights(sharpes)
    rows = []
    for strat, w in zip(strategies, weights):
        kf = kelly_fraction(
            strat.get("win_rate", 0.5),
            strat.get("avg_win", 1.0),
            strat.get("avg_loss", 1.0),
        )
        allocated = total_capital * w
        rec = "ALLOCATE" if strat.get("sharpe", 0) >= 1.0 else "WATCH" if strat.get("sharpe", 0) >= 0.5 else "SKIP"
        rows.append({
            "strategy": strat["name"],
            "sharpe": strat.get("sharpe", 0),
            "weight_%": round(w * 100, 1),
            "allocated_$": round(allocated, 0),
            "kelly_f_%": round(kf * 100, 1),
            "recommendation": rec,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mean-Variance Optimization (Markowitz, analytical)
# ---------------------------------------------------------------------------

def markowitz_weights(
    returns_df: pd.DataFrame,
    risk_free: float = 0.0,
    n_portfolios: int = 500,
) -> dict:
    """
    Monte Carlo simulation of efficient frontier.
    Returns: weights for max-Sharpe portfolio, min-vol portfolio, and the frontier.
    returns_df: DataFrame of daily returns, one column per asset.
    """
    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    n = len(mu)
    rng = np.random.default_rng(42)

    results = []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - risk_free) / vol if vol > 0 else 0
        results.append({"weights": w, "return": ret, "vol": vol, "sharpe": sharpe})

    df = pd.DataFrame(results)
    max_sharpe_idx = df["sharpe"].idxmax()
    min_vol_idx = df["vol"].idxmin()

    cols = returns_df.columns.tolist()
    max_sharpe_w = {cols[i]: round(float(df.loc[max_sharpe_idx, "weights"][i]), 4) for i in range(n)}
    min_vol_w = {cols[i]: round(float(df.loc[min_vol_idx, "weights"][i]), 4) for i in range(n)}

    return {
        "max_sharpe": {
            "weights": max_sharpe_w,
            "return": round(df.loc[max_sharpe_idx, "return"], 4),
            "vol": round(df.loc[max_sharpe_idx, "vol"], 4),
            "sharpe": round(df.loc[max_sharpe_idx, "sharpe"], 3),
        },
        "min_vol": {
            "weights": min_vol_w,
            "return": round(df.loc[min_vol_idx, "return"], 4),
            "vol": round(df.loc[min_vol_idx, "vol"], 4),
            "sharpe": round(df.loc[min_vol_idx, "sharpe"], 3),
        },
        "frontier_df": df[["return", "vol", "sharpe"]],
    }
