"""
Walk-forward analysis, Monte Carlo acceptance test, and Strategy Audit.

Strategy Audit implements the 6-test framework from jackson-video-resources/skills
(strategy-audit skill, mode 1 — Full Strategy Stress Test):
  1. In-sample performance
  2. Walk-forward (OOS)
  3. Monte Carlo (shuffle)
  4. Parameter sensitivity (±20% on key params)
  5. Slippage & fee stress test (2×, 5× costs)
  6. Drawdown analysis
"""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class WalkForward:
    def __init__(self, n_splits: int = 5, train_pct: float = 0.7):
        self.n_splits = n_splits
        self.train_pct = train_pct

    def split(self, index: pd.Index) -> Generator[tuple[pd.Index, pd.Index], None, None]:
        """Yield (train_index, test_index) pairs — no overlap, no future leak."""
        n = len(index)
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size
            if end > n:
                break
            split_point = start + int(fold_size * self.train_pct)
            train_idx = index[start:split_point]
            test_idx = index[split_point:end]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def run_walk_forward(
    df: pd.DataFrame,
    strategy_fn,
    strategy_params: dict,
    n_splits: int = 5,
    train_pct: float = 0.7,
    starting_capital: float = 100_000,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Run walk-forward validation.

    strategy_fn: callable(df_slice, **params) -> pd.Series of signals
    Returns DataFrame with columns: fold, train_sharpe, test_sharpe, n_trades
    """
    from engine.backtest import Backtest

    wf = WalkForward(n_splits, train_pct)
    rows = []

    for fold_i, (train_idx, test_idx) in enumerate(wf.split(df.index)):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        try:
            train_signals = strategy_fn(train_df, **strategy_params)
            if isinstance(train_signals, pd.DataFrame):
                train_signals = train_signals["signal"]
            train_bt = Backtest(train_df, train_signals, {}, starting_capital, commission_pct, slippage_pct)
            train_result = train_bt.run()
            train_sharpe = train_result.metrics.get("sharpe", np.nan)
        except Exception:
            train_sharpe = np.nan

        try:
            test_signals = strategy_fn(test_df, **strategy_params)
            if isinstance(test_signals, pd.DataFrame):
                test_signals = test_signals["signal"]
            test_bt = Backtest(test_df, test_signals, {}, starting_capital, commission_pct, slippage_pct)
            test_result = test_bt.run()
            test_sharpe = test_result.metrics.get("sharpe", np.nan)
            n_trades = test_result.metrics.get("n_trades", 0)
        except Exception:
            test_sharpe = np.nan
            n_trades = 0

        rows.append({
            "fold": fold_i + 1,
            "train_start": str(train_idx[0].date()),
            "train_end": str(train_idx[-1].date()),
            "test_start": str(test_idx[0].date()),
            "test_end": str(test_idx[-1].date()),
            "train_sharpe": round(train_sharpe, 3) if not np.isnan(train_sharpe) else None,
            "test_sharpe": round(test_sharpe, 3) if not np.isnan(test_sharpe) else None,
            "n_trades": n_trades,
        })

    return pd.DataFrame(rows)


def plot_walk_forward(wf_df: pd.DataFrame) -> go.Figure:
    """Bar chart of train vs. test Sharpe by fold."""
    fig = go.Figure()
    fig.add_bar(
        x=wf_df["fold"].astype(str),
        y=wf_df["train_sharpe"],
        name="Train Sharpe",
        marker_color="steelblue",
    )
    fig.add_bar(
        x=wf_df["fold"].astype(str),
        y=wf_df["test_sharpe"],
        name="Test Sharpe (OOS)",
        marker_color="darkorange",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Min OOS target")
    fig.update_layout(
        title="Walk-Forward: Train vs Out-of-Sample Sharpe",
        xaxis_title="Fold", yaxis_title="Sharpe Ratio",
        barmode="group", height=350, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo acceptance test
# ---------------------------------------------------------------------------

def monte_carlo_test(
    trades_pnl: pd.Series,
    n_sims: int = 1_000,
    starting_capital: float = 100_000,
) -> dict:
    """
    Shuffle trade P&L n_sims times and compute Sharpe for each path.
    Acceptance rule: real_sharpe >= median(sim_sharpes)
                 AND real_sharpe <= percentile(sim_sharpes, 95)

    Returns dict with: verdict, real_sharpe, median_sim, p95_sim,
                       sim_sharpes (array), p_value
    """
    if trades_pnl is None or len(trades_pnl) < 5:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "real_sharpe": np.nan,
            "median_sim": np.nan,
            "p95_sim": np.nan,
            "sim_sharpes": np.array([]),
            "p_value": np.nan,
        }

    pnl = trades_pnl.dropna().values
    real_sharpe = _sharpe_from_pnl(pnl)

    sim_sharpes = []
    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled = rng.permutation(pnl)
        sim_sharpes.append(_sharpe_from_pnl(shuffled))

    sim_sharpes = np.array(sim_sharpes)
    median_sim = float(np.median(sim_sharpes))
    p95_sim = float(np.percentile(sim_sharpes, 95))
    p_value = float((sim_sharpes >= real_sharpe).mean())

    if real_sharpe >= median_sim and real_sharpe <= p95_sim:
        verdict = "PASS"
    elif real_sharpe > p95_sim:
        verdict = "FAIL_OVERFIT"
    else:
        verdict = "FAIL_UNDERPERFORM"

    return {
        "verdict": verdict,
        "real_sharpe": round(real_sharpe, 3),
        "median_sim": round(median_sim, 3),
        "p95_sim": round(p95_sim, 3),
        "sim_sharpes": sim_sharpes,
        "p_value": round(p_value, 4),
    }


def _sharpe_from_pnl(pnl: np.ndarray) -> float:
    """Annualized Sharpe from a sequence of trade P&L percentages (no time adjustment — order-shuffled)."""
    if len(pnl) == 0:
        return 0.0
    mu = np.mean(pnl)
    sigma = np.std(pnl)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(252))


def plot_monte_carlo(mc_result: dict) -> go.Figure:
    """Histogram of simulated Sharpes with real Sharpe annotated."""
    sim = mc_result.get("sim_sharpes", np.array([]))
    real = mc_result.get("real_sharpe", np.nan)

    fig = go.Figure()
    if len(sim) > 0:
        fig.add_trace(go.Histogram(
            x=sim, nbinsx=50,
            name="Simulated Sharpes",
            marker_color="steelblue", opacity=0.7,
        ))

    if not np.isnan(real):
        fig.add_vline(
            x=real, line_color="red", line_dash="solid", line_width=2,
            annotation_text=f"Real: {real:.2f}",
            annotation_position="top right",
        )

    p95 = mc_result.get("p95_sim", np.nan)
    med = mc_result.get("median_sim", np.nan)
    if not np.isnan(p95):
        fig.add_vline(x=p95, line_color="orange", line_dash="dash", annotation_text="95th pct")
    if not np.isnan(med):
        fig.add_vline(x=med, line_color="green", line_dash="dash", annotation_text="Median")

    verdict = mc_result.get("verdict", "")
    fig.update_layout(
        title=f"Monte Carlo Acceptance Test — {verdict}",
        xaxis_title="Sharpe Ratio", yaxis_title="Count",
        height=350, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 6-Test Strategy Audit (skills/strategy-audit, mode 1)
# ---------------------------------------------------------------------------

def strategy_audit(
    df: pd.DataFrame,
    signal: pd.Series,
    strategy_fn=None,
    strategy_params: dict | None = None,
    starting_capital: float = 100_000,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.001,
    param_sensitivity_pct: float = 0.20,
) -> dict:
    """
    Run the 6-test strategy stress test framework.

    Tests:
      1. In-sample performance metrics
      2. Walk-forward OOS Sharpe (5 folds)
      3. Monte Carlo shuffle (1,000 sims)
      4. Parameter sensitivity (±20% on commission and slippage as proxies)
      5. Slippage & fee stress (2× and 5× cost)
      6. Drawdown analysis (max DD, recovery time, DD distribution)

    Pass/fail thresholds (from skills repo):
      Sharpe > 1.0 OOS ✅
      Max drawdown < 20% ✅
      Monte Carlo PASS ✅
      Param sensitivity: Sharpe drop < 30% ✅
      Min 50 trades ✅
    """
    from engine.backtest import Backtest

    results = {}

    # ---- Test 1: In-sample -------------------------------------------------
    bt = Backtest(df, signal, {}, starting_capital, commission_pct, slippage_pct)
    result = bt.run()
    results["in_sample"] = result.metrics
    results["in_sample"]["pass"] = (
        result.metrics.get("sharpe", 0) > 1.0 and
        abs(result.metrics.get("max_drawdown", -1)) < 0.20 and
        result.metrics.get("n_trades", 0) >= 50
    )

    # ---- Test 2: Walk-forward ----------------------------------------------
    if strategy_fn is not None and strategy_params is not None:
        wf_df = run_walk_forward(
            df, strategy_fn, strategy_params, n_splits=5, train_pct=0.7,
            starting_capital=starting_capital,
            commission_pct=commission_pct, slippage_pct=slippage_pct,
        )
        mean_oos = wf_df["test_sharpe"].dropna().mean()
        results["walk_forward"] = {
            "folds": wf_df.to_dict("records"),
            "mean_oos_sharpe": round(float(mean_oos), 3) if not np.isnan(mean_oos) else None,
            "pass": bool(mean_oos >= 1.0) if not np.isnan(mean_oos) else False,
        }
    else:
        results["walk_forward"] = {"pass": None, "note": "No strategy_fn provided"}

    # ---- Test 3: Monte Carlo -----------------------------------------------
    pnl = result.trades["pnl_pct"] if not result.trades.empty else pd.Series(dtype=float)
    mc = monte_carlo_test(pnl, n_sims=1_000)
    results["monte_carlo"] = mc
    results["monte_carlo"]["pass"] = mc["verdict"] == "PASS"

    # ---- Test 4: Parameter sensitivity (±20% on costs as proxy) -----------
    base_sharpe = result.metrics.get("sharpe", 0)
    sensitivity_results = []
    for delta in [-0.20, +0.20]:
        adj_comm = commission_pct * (1 + delta)
        adj_slip = slippage_pct * (1 + delta)
        bt_adj = Backtest(df, signal, {}, starting_capital, adj_comm, adj_slip)
        r_adj = bt_adj.run()
        adj_sharpe = r_adj.metrics.get("sharpe", 0)
        drop = (base_sharpe - adj_sharpe) / base_sharpe if base_sharpe != 0 else 0
        sensitivity_results.append({
            "delta": f"{delta:+.0%}",
            "sharpe": round(adj_sharpe, 3),
            "sharpe_drop": round(drop, 3),
        })
    max_drop = max(abs(r["sharpe_drop"]) for r in sensitivity_results)
    results["parameter_sensitivity"] = {
        "results": sensitivity_results,
        "max_sharpe_drop": round(max_drop, 3),
        "pass": max_drop < 0.30,
    }

    # ---- Test 5: Cost stress test (2× and 5× costs) -----------------------
    stress = []
    for mult in [2, 5]:
        bt_s = Backtest(df, signal, {}, starting_capital, commission_pct * mult, slippage_pct * mult)
        r_s = bt_s.run()
        stress.append({
            "multiplier": f"{mult}×",
            "sharpe": round(r_s.metrics.get("sharpe", 0), 3),
            "max_dd": round(r_s.metrics.get("max_drawdown", 0) * 100, 1),
        })
    results["cost_stress"] = {
        "results": stress,
        "pass": all(r["sharpe"] > 0 for r in stress),
    }

    # ---- Test 6: Drawdown analysis -----------------------------------------
    dd = result.drawdown
    dd_pct = dd * 100
    # Recovery times: count bars in each drawdown
    in_dd = (dd < -0.01)
    dd_periods = []
    start = None
    for i, (ts, v) in enumerate(in_dd.items()):
        if v and start is None:
            start = i
        elif not v and start is not None:
            dd_periods.append(i - start)
            start = None
    results["drawdown_analysis"] = {
        "max_drawdown_pct": round(float(dd_pct.min()), 2),
        "avg_drawdown_pct": round(float(dd_pct[dd_pct < 0].mean()), 2) if (dd_pct < 0).any() else 0,
        "n_drawdown_periods": len(dd_periods),
        "avg_recovery_bars": round(float(np.mean(dd_periods)), 1) if dd_periods else 0,
        "max_recovery_bars": int(max(dd_periods)) if dd_periods else 0,
        "pass": abs(float(dd_pct.min())) < 20,
    }

    # ---- Overall verdict ---------------------------------------------------
    tests_passed = sum([
        results["in_sample"].get("pass", False),
        results.get("walk_forward", {}).get("pass") or False,
        results["monte_carlo"].get("pass", False),
        results["parameter_sensitivity"].get("pass", False),
        results["cost_stress"].get("pass", False),
        results["drawdown_analysis"].get("pass", False),
    ])
    results["overall"] = {
        "tests_passed": tests_passed,
        "tests_total": 6,
        "verdict": "STRONG" if tests_passed >= 5 else "MARGINAL" if tests_passed >= 3 else "FAIL",
    }

    return results
