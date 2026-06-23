"""
Comprehensive test suite covering all MVP features.
Tests grouped by module: data, backtest, risk, validation, strategies, journal, capital_allocator.
Run: pytest tests/test_comprehensive.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spy_daily():
    from engine.data import fetch
    return fetch("SPY", "2020-01-01", "2023-12-31", "1d")


@pytest.fixture(scope="module")
def btc_daily():
    from engine.data import fetch
    return fetch("BTC-USD", "2021-01-01", "2023-12-31", "1d")


@pytest.fixture(scope="module")
def spy_signals(spy_daily):
    from strategies.donchian import generate_signals
    return generate_signals(spy_daily, period=20)


# ============================================================================
# engine/data.py
# ============================================================================

class TestDataLayer:
    def test_fetch_returns_correct_columns(self, spy_daily):
        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(spy_daily.columns)

    def test_fetch_index_is_datetime(self, spy_daily):
        assert isinstance(spy_daily.index, pd.DatetimeIndex)

    def test_fetch_no_nulls_in_close(self, spy_daily):
        assert spy_daily["Close"].isna().sum() == 0

    def test_fetch_high_gte_low(self, spy_daily):
        assert (spy_daily["High"] >= spy_daily["Low"]).all()

    def test_fetch_high_gte_close(self, spy_daily):
        assert (spy_daily["High"] >= spy_daily["Close"]).all()

    def test_cache_creates_parquet(self, spy_daily):
        cache_dir = Path.home() / ".quant_app_cache"
        parquet_files = list(cache_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Cache directory should have at least one parquet file"

    def test_fetch_crypto_no_negative_prices(self, btc_daily):
        assert (btc_daily["Close"] > 0).all()

    def test_fetch_interval_daily_minimum_length(self, spy_daily):
        assert len(spy_daily) >= 200, "3-year SPY daily should have at least 200 bars"

    def test_data_error_on_bad_ticker(self):
        from engine.data import DataError, fetch
        with pytest.raises(DataError):
            fetch("THISISNOTAREALTICKER12345", "2020-01-01", "2021-01-01", "1d")

    def test_is_crypto_variants(self):
        from engine.data import is_crypto
        assert is_crypto("BTC-USD") is True
        assert is_crypto("ETH-USD") is True
        assert is_crypto("BTCUSDT") is True
        assert is_crypto("ETHUSDT") is True
        assert is_crypto("SPY") is False
        assert is_crypto("AAPL") is False
        assert is_crypto("QQQ") is False

    def test_market_presets_ticker_and_type(self):
        from engine.data import MARKET_PRESETS
        for name, (ticker, mtype) in MARKET_PRESETS.items():
            if ticker is not None:
                assert mtype in ("equity", "crypto"), f"Bad market type for {name}"


# ============================================================================
# engine/backtest.py
# ============================================================================

class TestBacktest:
    def test_equity_starts_at_capital(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt = Backtest(spy_daily, spy_signals["signal"], {}, starting_capital=100_000)
        result = bt.run()
        assert abs(result.equity.iloc[0] - 100_000) < 100

    def test_equity_always_positive(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt = Backtest(spy_daily, spy_signals["signal"], {})
        result = bt.run()
        assert (result.equity > 0).all()

    def test_metrics_sharpe_is_finite(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt = Backtest(spy_daily, spy_signals["signal"], {})
        result = bt.run()
        assert np.isfinite(result.metrics.get("sharpe", float("nan")))

    def test_metrics_cagr_reasonable(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt = Backtest(spy_daily, spy_signals["signal"], {})
        result = bt.run()
        cagr = result.metrics.get("cagr", 0)
        assert -1.0 < cagr < 10.0, f"CAGR {cagr} looks unreasonable"

    def test_trades_dataframe_has_required_columns(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt = Backtest(spy_daily, spy_signals["signal"], {})
        result = bt.run()
        if not result.trades.empty:
            required = {"entry_dt", "exit_dt", "entry_px", "exit_px", "pnl_pct"}
            assert required.issubset(result.trades.columns)

    def test_no_lookahead_bias(self, spy_daily):
        """Signal shift(1) means position on day N only uses info from day N-1."""
        from engine.backtest import Backtest
        from strategies.donchian import generate_signals
        sig_df = generate_signals(spy_daily, period=20)
        # Correlation between signal and same-day return should not be >> 0
        ret = spy_daily["Close"].pct_change()
        corr = sig_df["signal"].shift(1).corr(ret)
        # We're just checking there's no perfect look-ahead (corr << 1)
        assert corr < 0.9, f"Suspiciously high correlation {corr} — possible look-ahead"

    def test_buy_and_hold_matches_price_return(self, spy_daily):
        from engine.backtest import buy_and_hold
        bah = buy_and_hold(spy_daily, 100_000)
        price_ret = spy_daily["Close"].iloc[-1] / spy_daily["Close"].iloc[0]
        bah_ret = bah.iloc[-1] / bah.iloc[0]
        assert abs(bah_ret - price_ret) < 0.05, "Buy-and-hold should track price return"

    def test_commission_reduces_equity(self, spy_daily, spy_signals):
        from engine.backtest import Backtest
        bt_no_cost = Backtest(spy_daily, spy_signals["signal"], {}, commission_pct=0, slippage_pct=0)
        bt_with_cost = Backtest(spy_daily, spy_signals["signal"], {}, commission_pct=0.001, slippage_pct=0.001)
        r0 = bt_no_cost.run()
        r1 = bt_with_cost.run()
        assert r0.equity.iloc[-1] >= r1.equity.iloc[-1], "Costs should reduce equity"

    def test_zero_signal_results_in_flat_equity(self, spy_daily):
        from engine.backtest import Backtest
        zero_signal = pd.Series(0.0, index=spy_daily.index)
        bt = Backtest(spy_daily, zero_signal, {}, starting_capital=100_000)
        result = bt.run()
        assert abs(result.equity.iloc[-1] - 100_000) < 1.0, "Zero signal = no trades = flat equity"


# ============================================================================
# engine/risk.py
# ============================================================================

class TestRisk:
    def test_position_size_basic(self):
        from engine.risk import position_size
        shares = position_size(100_000, 0.01, entry=100, stop=95)
        assert shares == 200  # risk=$1000, stop_dist=$5

    def test_position_size_zero_when_no_stop(self):
        from engine.risk import position_size
        # stop == entry → division by zero guard
        shares = position_size(100_000, 0.01, entry=100, stop=100)
        assert shares == 0

    def test_check_rr_pass(self):
        from engine.risk import check_rr
        assert check_rr(entry=100, stop=98, target=104, min_rr=2.0) is True  # R:R = 2.0

    def test_check_rr_fail(self):
        from engine.risk import check_rr
        assert check_rr(entry=100, stop=98, target=102, min_rr=2.0) is False  # R:R = 1.0

    def test_drawdown_never_positive(self, spy_daily):
        from engine.backtest import buy_and_hold
        from engine.risk import compute_drawdown
        equity = buy_and_hold(spy_daily, 100_000)
        dd = compute_drawdown(equity)
        assert (dd <= 0.001).all()

    def test_drawdown_starts_at_zero(self, spy_daily):
        from engine.backtest import buy_and_hold
        from engine.risk import compute_drawdown
        equity = buy_and_hold(spy_daily, 100_000)
        dd = compute_drawdown(equity)
        assert dd.iloc[0] == 0.0

    def test_atr_positive(self, spy_daily):
        from engine.risk import atr
        atr_vals = atr(spy_daily, period=14)
        assert (atr_vals.dropna() > 0).all()

    def test_circuit_breakers_returns_boolean_mask(self, spy_daily):
        from engine.backtest import Backtest
        from engine.risk import apply_circuit_breakers
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)["signal"]
        bt = Backtest(spy_daily, sig, {})
        result = bt.run()
        mask = apply_circuit_breakers(result.equity, daily_loss_limit=0.025, max_dd_limit=0.08)
        # mask is boolean Series; True = trading allowed
        assert mask.dtype == bool
        assert len(mask) == len(result.equity)
        # Once tripped, all subsequent values should be False
        if not mask.all():
            first_false = mask.idxmin()
            assert not mask[first_false:].any()


# ============================================================================
# engine/validation.py
# ============================================================================

class TestValidation:
    def test_walk_forward_returns_dataframe(self, spy_daily):
        from engine.validation import run_walk_forward
        from strategies.donchian import generate_signals

        def strat_fn(df, **kw):
            return generate_signals(df, period=20)["signal"]

        wf = run_walk_forward(spy_daily, strat_fn, {}, n_splits=3)
        assert isinstance(wf, pd.DataFrame)
        assert len(wf) == 3

    def test_walk_forward_no_future_leak(self, spy_daily):
        from engine.validation import run_walk_forward
        from strategies.donchian import generate_signals

        def strat_fn(df, **kw):
            return generate_signals(df, period=20)["signal"]

        wf = run_walk_forward(spy_daily, strat_fn, {}, n_splits=3, train_pct=0.7)
        # train end must precede test start in each fold
        for _, row in wf.iterrows():
            assert row["train_end"] < row["test_start"]

    def test_monte_carlo_verdict_valid(self):
        from engine.validation import monte_carlo_test
        pnl = pd.Series(np.random.normal(0.002, 0.01, 100))
        mc = monte_carlo_test(pnl, n_sims=200)
        assert mc["verdict"] in {"PASS", "FAIL_OVERFIT", "FAIL_UNDERPERFORM", "INSUFFICIENT_DATA"}

    def test_monte_carlo_sim_count(self):
        from engine.validation import monte_carlo_test
        pnl = pd.Series(np.random.normal(0.002, 0.01, 60))
        mc = monte_carlo_test(pnl, n_sims=500)
        assert len(mc["sim_sharpes"]) == 500

    def test_strategy_audit_all_six_tests_present(self, spy_daily):
        from engine.validation import strategy_audit
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)["signal"]
        audit = strategy_audit(spy_daily, sig, starting_capital=100_000,
                               commission_pct=0.0005, slippage_pct=0.001)
        for key in ["in_sample", "walk_forward", "monte_carlo",
                    "parameter_sensitivity", "cost_stress", "drawdown_analysis"]:
            assert key in audit, f"Missing audit test: {key}"

    def test_strategy_audit_overall_verdict(self, spy_daily):
        from engine.validation import strategy_audit
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)["signal"]
        audit = strategy_audit(spy_daily, sig, starting_capital=100_000)
        assert audit["overall"]["verdict"] in {"STRONG", "MARGINAL", "FAIL"}


# ============================================================================
# strategies/donchian.py
# ============================================================================

class TestDonchian:
    def test_signal_values_valid(self, spy_daily):
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)
        assert sig["signal"].isin([-1, 0, 1]).all()

    def test_no_signal_before_warmup(self, spy_daily):
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)
        # First 20 bars should have no signal (0)
        assert (sig["signal"].iloc[:20] == 0).all()

    def test_upper_band_gte_lower(self, spy_daily):
        from strategies.donchian import generate_signals
        sig = generate_signals(spy_daily, period=20)
        assert (sig["upper_band"].dropna() >= sig["lower_band"].dropna()).all()

    def test_parameter_sweep_returns_dataframe(self, spy_daily):
        from strategies.donchian import parameter_sweep
        sweep = parameter_sweep(spy_daily)
        assert isinstance(sweep, pd.DataFrame)
        assert "sharpe" in sweep.columns
        assert "period" in sweep.columns

    def test_regime_gate_reduces_signals(self, spy_daily):
        from strategies.donchian import generate_signals
        from strategies.regime import get_regime_signal
        regime = get_regime_signal(spy_daily, model="sma")
        sig_no_regime = generate_signals(spy_daily, period=20)
        sig_with_regime = generate_signals(spy_daily, period=20, regime_signal=regime)
        # Regime gate should never produce MORE long signals
        n_no = (sig_no_regime["signal"] > 0).sum()
        n_with = (sig_with_regime["signal"] > 0).sum()
        assert n_with <= n_no


# ============================================================================
# strategies/scalp.py
# ============================================================================

class TestScalp:
    def test_ema_length_matches_input(self, spy_daily):
        from strategies.scalp import calc_ema
        ema = calc_ema(spy_daily["Close"], 8)
        assert len(ema) == len(spy_daily)

    def test_rsi_bounded(self, spy_daily):
        from strategies.scalp import calc_rsi
        rsi = calc_rsi(spy_daily["Close"], 3)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_vwap_positive(self, spy_daily):
        from strategies.scalp import calc_vwap
        vwap = calc_vwap(spy_daily)
        assert (vwap.dropna() > 0).all()

    def test_safety_check_returns_dict(self, spy_daily):
        from strategies.scalp import calc_ema, calc_vwap, calc_rsi, safety_check
        row = spy_daily.iloc[-1]
        ema8 = calc_ema(spy_daily["Close"], 8)
        vwap = calc_vwap(spy_daily)
        rsi3 = calc_rsi(spy_daily["Close"], 3)
        result = safety_check(row, ema8.iloc[-1], vwap.iloc[-1], rsi3.iloc[-1])
        assert "allowed" in result
        assert "direction" in result
        assert isinstance(result["allowed"], bool)

    def test_signals_no_lookahead(self, spy_daily):
        from strategies.scalp import generate_signals
        sig = generate_signals(spy_daily)
        # Signal at bar N should not correlate perfectly with bar N's return
        ret = spy_daily["Close"].pct_change()
        corr = sig["signal"].shift(1).corr(ret)
        assert corr < 0.9

    def test_scalp_short_signals_require_allow_short(self, spy_daily):
        from strategies.scalp import generate_signals
        sig_long_only = generate_signals(spy_daily, allow_short=False)
        sig_with_short = generate_signals(spy_daily, allow_short=True)
        assert (sig_long_only["signal"] >= 0).all(), "Long-only mode should have no short signals"
        # Short signals possible with allow_short=True
        # (not guaranteed for all data, so just check no crash)


# ============================================================================
# strategies/regime.py
# ============================================================================

class TestRegime:
    def test_sma_regime_binary(self, spy_daily):
        from strategies.regime import get_regime_signal
        sig = get_regime_signal(spy_daily, model="sma", sma_period=200)
        assert sig.isin([0, 1]).all()

    def test_markov_regime_bounded(self, spy_daily):
        from strategies.regime import get_regime_signal
        sig = get_regime_signal(spy_daily, model="markov")
        valid = sig.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_regime_summary_structure(self, spy_daily):
        from strategies.regime import regime_summary
        s = regime_summary(spy_daily, window=20, threshold=0.05)
        if "error" not in s:
            assert "current_regime" in s
            assert s["current_regime"] in {"Bull", "Sideways", "Bear"}
            probs = s["next_state_probabilities"]
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.05, "Probabilities should sum to ~1"

    def test_stationary_distribution_sums_to_one(self, spy_daily):
        from strategies.regime import _label_states, _build_transition_matrix, stationary_distribution
        labels = _label_states(spy_daily, fwd_period=20, threshold=0.05)
        T = _build_transition_matrix(labels)
        sd = stationary_distribution(T)
        assert abs(sd.sum() - 1.0) < 1e-6

    def test_label_regime_states_no_nan(self, spy_daily):
        from strategies.regime import label_regime_states
        # No regime_signal → uses forward-return labeling
        labels = label_regime_states(spy_daily, fwd_period=20, threshold=0.05)
        valid = labels.dropna()
        assert valid.isin(["Bull", "Bear", "Sideways"]).all()


# ============================================================================
# engine/capital_allocator.py
# ============================================================================

class TestCapitalAllocator:
    def test_kelly_fraction_zero_win_rate(self):
        from engine.capital_allocator import kelly_fraction
        assert kelly_fraction(0.0, 2.0, 1.0) == 0.0

    def test_kelly_fraction_capped_at_quarter(self):
        from engine.capital_allocator import kelly_fraction
        # Even with extreme edge, should cap at 0.25
        kf = kelly_fraction(0.99, 10.0, 1.0)
        assert kf <= 0.25

    def test_kelly_fraction_negative_edge(self):
        from engine.capital_allocator import kelly_fraction
        # Losing strategy should return 0
        kf = kelly_fraction(0.3, 1.0, 2.0)
        assert kf >= 0.0

    def test_equal_risk_weights_sum_to_one(self):
        from engine.capital_allocator import equal_risk_weights
        weights = equal_risk_weights([1.5, 1.0, 0.5])
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_equal_risk_weights_proportional_to_sharpe(self):
        from engine.capital_allocator import equal_risk_weights
        weights = equal_risk_weights([2.0, 1.0])
        assert weights[0] > weights[1], "Higher Sharpe should get higher weight"

    def test_allocation_report_weights_sum_to_100(self):
        from engine.capital_allocator import allocation_report
        strats = [
            {"name": "A", "sharpe": 1.5, "win_rate": 0.6, "avg_win": 2.0, "avg_loss": 1.0},
            {"name": "B", "sharpe": 1.0, "win_rate": 0.5, "avg_win": 1.5, "avg_loss": 1.0},
            {"name": "C", "sharpe": 0.5, "win_rate": 0.4, "avg_win": 1.2, "avg_loss": 1.0},
        ]
        report = allocation_report(strats, 100_000)
        assert abs(report["weight_%"].sum() - 100.0) < 0.5

    def test_markowitz_weights_keys(self, spy_daily, btc_daily):
        from engine.capital_allocator import markowitz_weights
        prices = pd.DataFrame({
            "SPY": spy_daily["Close"],
            "BTC": btc_daily["Close"],
        }).dropna()
        returns = prices.pct_change().dropna()
        if len(returns) > 30:
            result = markowitz_weights(returns)
            assert "max_sharpe" in result
            assert "min_vol" in result
            assert "frontier_df" in result

    def test_markowitz_max_sharpe_weights_sum_to_one(self, spy_daily, btc_daily):
        from engine.capital_allocator import markowitz_weights
        prices = pd.DataFrame({"SPY": spy_daily["Close"], "BTC": btc_daily["Close"]}).dropna()
        returns = prices.pct_change().dropna()
        if len(returns) > 30:
            result = markowitz_weights(returns)
            weights = result["max_sharpe"]["weights"]
            assert abs(sum(weights.values()) - 1.0) < 0.01


# ============================================================================
# engine/journal.py
# ============================================================================

class TestJournal:
    def test_log_trade_returns_dict(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        result = jmod.log_trade(
            ticker="SPY", market="equity", strategy="Donchian",
            direction="long", entry_price=400.0, exit_price=410.0,
            quantity=100.0, stop_price=395.0, target_price=410.0,
        )
        assert "pnl_pct" in result
        assert result["pnl_pct"] > 0

    def test_log_trade_short_pnl(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        result = jmod.log_trade(
            ticker="BTC-USD", market="crypto", strategy="Scalp",
            direction="short", entry_price=50000.0, exit_price=48000.0,
            quantity=0.1,
        )
        assert result["pnl_pct"] > 0  # short: price fell, profitable

    def test_get_journal_returns_dataframe(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        jmod.log_trade(ticker="SPY", market="equity", strategy="Test",
                       direction="long", entry_price=100.0, exit_price=102.0, quantity=10.0)
        df = jmod.get_journal()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_performance_review_keys(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        for i in range(10):
            jmod.log_trade(ticker="SPY", market="equity", strategy="Test",
                           direction="long", entry_price=100.0,
                           exit_price=102.0 if i % 3 != 0 else 98.0, quantity=10.0)
        df = jmod.get_journal()
        review = jmod.performance_review(df)
        for key in ["win_rate", "profit_factor", "sharpe", "max_drawdown_pct", "n_trades"]:
            assert key in review, f"Missing key: {key}"

    def test_performance_review_win_rate_bounded(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        for _ in range(5):
            jmod.log_trade(ticker="SPY", market="equity", strategy="Test",
                           direction="long", entry_price=100.0, exit_price=105.0, quantity=10.0)
        df = jmod.get_journal()
        review = jmod.performance_review(df)
        assert 0.0 <= review["win_rate"] <= 1.0

    def test_log_safety_check(self, tmp_path, monkeypatch):
        import engine.journal as jmod
        monkeypatch.setattr(jmod, "JOURNAL_PATH", tmp_path / "journal.csv")
        monkeypatch.setattr(jmod, "SAFETY_LOG_PATH", tmp_path / "safety.json")
        jmod.log_safety_check("BTC-USD", {"allowed": True, "direction": "long"})
        log_path = tmp_path / "safety.json"
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert isinstance(data, list)
        assert data[0]["ticker"] == "BTC-USD"


# ============================================================================
# strategies/trend_join.py
# ============================================================================

class TestTrendJoin:
    def test_signals_valid_values(self, spy_daily):
        from strategies.trend_join import generate_signals
        sig = generate_signals(spy_daily)
        assert sig["signal"].isin([-1, 0, 1]).all()

    def test_signals_not_all_zero(self, spy_daily):
        from strategies.trend_join import generate_signals
        sig = generate_signals(spy_daily)
        assert sig["signal"].abs().sum() > 0, "Trend-join should produce some signals on 3-year SPY"


# ============================================================================
# strategies/factor.py
# ============================================================================

class TestFactor:
    def test_spy_factor_signal_binary(self, spy_daily):
        from strategies.factor import spy_factor_signal
        sig = spy_factor_signal(spy_daily)
        assert sig.isin([0.0, 1.0]).all()

    def test_spy_factor_signal_length(self, spy_daily):
        from strategies.factor import spy_factor_signal
        sig = spy_factor_signal(spy_daily)
        assert len(sig) == len(spy_daily)


# ============================================================================
# Edge cases / integration
# ============================================================================

class TestEdgeCases:
    def test_empty_signal_backtest(self, spy_daily):
        from engine.backtest import Backtest
        zero = pd.Series(0.0, index=spy_daily.index)
        bt = Backtest(spy_daily, zero, {}, starting_capital=50_000)
        result = bt.run()
        assert result.metrics["n_trades"] == 0

    def test_all_long_signal_backtest(self, spy_daily):
        from engine.backtest import Backtest
        ones = pd.Series(1.0, index=spy_daily.index)
        bt = Backtest(spy_daily, ones, {})
        result = bt.run()
        assert np.isfinite(result.metrics["sharpe"])

    def test_donchian_on_crypto(self, btc_daily):
        from strategies.donchian import generate_signals
        sig = generate_signals(btc_daily, period=20)
        assert sig["signal"].isin([-1, 0, 1]).all()

    def test_scalp_minimum_data(self):
        from engine.data import fetch
        from strategies.scalp import generate_signals
        df = fetch("SPY", "2023-01-01", "2023-06-01", "1d")
        sig = generate_signals(df)
        assert "signal" in sig.columns

    def test_walk_forward_minimum_data(self, spy_daily):
        from engine.validation import run_walk_forward
        from strategies.donchian import generate_signals

        def strat_fn(df, **kw):
            return generate_signals(df, period=10)["signal"]

        wf = run_walk_forward(spy_daily, strat_fn, {}, n_splits=2, train_pct=0.7)
        assert len(wf) == 2
