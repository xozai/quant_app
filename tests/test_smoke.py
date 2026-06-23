"""Smoke tests — run with: pytest tests/ -v"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data import fetch, is_crypto, MARKET_PRESETS
from engine.backtest import Backtest, rolling_sharpe, benchmark_metrics
from engine.risk import atr, compute_drawdown
from engine.scanner import scan_universe
from engine.validation import monte_carlo_test, strategy_audit
from engine.capital_allocator import kelly_fraction, allocation_report
from engine.journal import performance_review
from strategies.regime import get_regime_signal, label_regime_states, regime_summary
from strategies.donchian import generate_signals as donchian_signals
from strategies.scalp import generate_signals as scalp_signals, calc_ema, calc_rsi, calc_vwap


TICKER = "SPY"
START = "2020-01-01"
END = "2023-12-31"


@pytest.fixture(scope="module")
def spy_daily():
    """Fetch (or load cached) SPY daily data."""
    return fetch(TICKER, START, END, "1d")


def test_data_fetch_returns_ohlcv(spy_daily):
    df = spy_daily
    assert not df.empty, "DataFrame should not be empty"
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns, f"Missing column: {col}"
    assert df.index.is_monotonic_increasing, "Index should be sorted"
    assert len(df) > 100, "Expected > 100 bars of SPY data"


def test_donchian_backtest_runs(spy_daily):
    df = spy_daily
    signals = donchian_signals(df, period=20, atr_period=14, atr_k=2.0)
    assert "signal" in signals.columns
    bt = Backtest(df, signals["signal"], {}, starting_capital=100_000)
    result = bt.run()
    assert np.isfinite(result.metrics["sharpe"]), "Sharpe should be finite"
    assert result.equity.iloc[-1] > 0, "Final equity should be positive"
    assert 0 <= result.metrics["win_rate"] <= 1, "Win rate should be in [0, 1]"


def test_regime_labels_cover_full_range(spy_daily):
    df = spy_daily
    regime = get_regime_signal(df, model="sma", sma_period=200)
    states = label_regime_states(df, regime)
    assert set(states.unique()).issubset({"Bull", "Sideways", "Bear"}), "States should only be Bull/Sideways/Bear"
    # After warm-up period, regime values should not all be NaN
    non_nan = regime.dropna()
    assert len(non_nan) > 50, "Should have > 50 valid regime values after warm-up"


def test_monte_carlo_returns_pass_or_fail(spy_daily):
    df = spy_daily
    signals = donchian_signals(df, period=20)
    bt = Backtest(df, signals["signal"], {}, starting_capital=100_000)
    result = bt.run()
    pnl = result.trades["pnl_pct"] if not result.trades.empty else pd.Series(dtype=float)
    mc = monte_carlo_test(pnl, n_sims=200)
    assert "verdict" in mc, "Result should have 'verdict' key"
    assert mc["verdict"] in {"PASS", "FAIL_OVERFIT", "FAIL_UNDERPERFORM", "INSUFFICIENT_DATA"}
    assert "real_sharpe" in mc
    assert "sim_sharpes" in mc


def test_atr_no_nan_after_warmup(spy_daily):
    df = spy_daily
    atr_vals = atr(df, period=14)
    post_warmup = atr_vals.iloc[14:]
    assert post_warmup.notna().all(), "ATR should have no NaN values after warm-up period"


def test_drawdown_non_positive(spy_daily):
    df = spy_daily
    equity = pd.Series(100_000 * (1 + df["Close"].pct_change().fillna(0)).cumprod())
    dd = compute_drawdown(equity)
    assert (dd <= 0.001).all(), "Drawdown should always be <= 0"


# ---------------------------------------------------------------------------
# New: NASDAQ + Crypto + Scalp + Capital Allocator + Regime Summary
# ---------------------------------------------------------------------------

def test_crypto_ticker_detection():
    assert is_crypto("BTC-USD") is True
    assert is_crypto("ETH-USD") is True
    assert is_crypto("BTCUSDT") is True
    assert is_crypto("SPY") is False
    assert is_crypto("QQQ") is False


def test_market_presets_complete():
    required = {"S&P 500 — SPY", "NASDAQ 100 — QQQ", "Bitcoin — BTC-USD"}
    assert required.issubset(set(MARKET_PRESETS.keys()))


@pytest.fixture(scope="module")
def btc_daily():
    return fetch("BTC-USD", "2022-01-01", "2024-06-01", "1d")


def test_crypto_data_fetch(btc_daily):
    df = btc_daily
    assert not df.empty, "BTC-USD data should not be empty"
    assert "Close" in df.columns
    assert len(df) > 100


def test_scalp_signals_run(spy_daily):
    df = spy_daily
    sigs = scalp_signals(df, ema_period=8, rsi_period=3)
    assert "signal" in sigs.columns
    assert "vwap" in sigs.columns
    assert "rsi3" in sigs.columns
    # Signal values should be -1, 0, or 1
    assert sigs["signal"].isin([-1, 0, 1]).all()


def test_scalp_signals_on_crypto(btc_daily):
    df = btc_daily
    sigs = scalp_signals(df, ema_period=8, rsi_period=3, allow_short=True)
    assert not sigs.empty
    assert "signal" in sigs.columns


def test_vwap_no_nan_after_warmup(spy_daily):
    vwap = calc_vwap(spy_daily)
    post = vwap.iloc[20:]
    assert post.notna().mean() > 0.95, "VWAP should have < 5% NaN after warm-up"


def test_kelly_fraction_reasonable():
    kf = kelly_fraction(win_rate=0.55, avg_win=2.0, avg_loss=1.0)
    assert 0 < kf <= 0.25, f"Kelly fraction {kf} outside (0, 0.25]"
    # Edge: zero win rate → should return 0
    assert kelly_fraction(0.0, 2.0, 1.0) == 0.0


def test_allocation_report_columns():
    strats = [
        {"name": "A", "sharpe": 1.5, "win_rate": 0.6, "avg_win": 2.0, "avg_loss": 1.0},
        {"name": "B", "sharpe": 0.8, "win_rate": 0.5, "avg_win": 1.5, "avg_loss": 1.0},
    ]
    report = allocation_report(strats, 100_000)
    assert set(["strategy", "sharpe", "weight_%", "allocated_$", "kelly_f_%", "recommendation"]).issubset(set(report.columns))
    assert abs(report["weight_%"].sum() - 100) < 1.0, "Weights should sum to ~100%"


def test_regime_summary_keys(spy_daily):
    summary = regime_summary(spy_daily, window=20, threshold=0.05)
    if "error" not in summary:
        for key in ["current_regime", "next_state_probabilities", "signal", "transition_matrix",
                    "stationary_distribution", "backtest"]:
            assert key in summary, f"Missing key: {key}"
        assert summary["current_regime"] in {"Bull", "Sideways", "Bear"}


def test_performance_review_empty():
    result = performance_review(pd.DataFrame())
    assert "error" in result, "Empty journal should return error dict"


def test_strategy_audit_structure(spy_daily):
    sigs = donchian_signals(spy_daily, period=20)
    audit = strategy_audit(
        spy_daily, sigs["signal"],
        starting_capital=100_000,
        commission_pct=0.0005,
        slippage_pct=0.001,
    )
    assert "overall" in audit
    assert audit["overall"]["tests_total"] == 6
    assert audit["overall"]["verdict"] in {"STRONG", "MARGINAL", "FAIL"}


def test_rolling_sharpe_length(spy_daily):
    from engine.backtest import Backtest
    from strategies.donchian import generate_signals
    sigs = generate_signals(spy_daily, period=20)
    bt = Backtest(spy_daily, sigs["signal"], {})
    r = bt.run()
    rs = rolling_sharpe(r.returns, window=63)
    assert len(rs) == len(r.returns), "Rolling Sharpe must match returns length"
    assert rs.dropna().between(-50, 50).all(), "Rolling Sharpe should be in a sane range"


def test_benchmark_metrics_keys(spy_daily):
    from engine.backtest import Backtest
    from strategies.donchian import generate_signals
    sigs = generate_signals(spy_daily, period=20)
    bt = Backtest(spy_daily, sigs["signal"], {})
    r = bt.run()
    spy_ret = spy_daily["Close"].pct_change().dropna()
    bm = benchmark_metrics(r.returns, spy_ret)
    if "error" not in bm:
        for key in ["alpha", "beta", "correlation", "information_ratio", "tracking_error"]:
            assert key in bm, f"Missing benchmark metric: {key}"
        assert -5 <= bm["beta"] <= 5, "Beta should be sane"


def test_scanner_returns_dataframe(spy_daily):
    import strategies.donchian as donchian_strat
    tickers = ["SPY", "QQQ"]
    start = str(spy_daily.index[0].date())
    end = str(spy_daily.index[-1].date())
    results = scan_universe(
        tickers,
        lambda df: donchian_strat.generate_signals(df, period=20)["signal"],
        {},
        start, end,
    )
    assert isinstance(results, pd.DataFrame), "Scanner must return a DataFrame"
    assert "ticker" in results.columns
    assert len(results) <= len(tickers)
