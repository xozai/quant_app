"""
Quant Trading App — Streamlit entry point.
Run: streamlit run app.py

Markets: S&P 500 / NASDAQ 100 / Crypto (BTC, ETH, SOL, …)
Strategies: Donchian Breakout | Trend-Join | VWAP+EMA+RSI Scalp | Factor | Regime Only
Validation: Walk-Forward | Monte Carlo | 6-Test Strategy Audit
Extras:  Regime Dashboard | Capital Allocator | Trade Journal | Agent Firm Spec
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data import (
    fetch, fetch_daily_for_regime, DataError,
    MARKET_PRESETS, CRYPTO_TICKERS, NASDAQ_100, is_crypto,
)
from engine.backtest import Backtest, buy_and_hold, rolling_sharpe, benchmark_metrics
from engine.risk import compute_drawdown
from engine.scanner import scan_universe
from engine.report import generate_html_report
from engine.validation import (
    run_walk_forward, monte_carlo_test, plot_monte_carlo,
    plot_walk_forward, strategy_audit,
)
from engine.capital_allocator import kelly_fraction, allocation_report, markowitz_weights
from engine.journal import log_trade, get_journal, performance_review
from strategies.regime import (
    get_regime_signal, plot_regime, regime_summary,
    label_regime_states,
)
import strategies.donchian as donchian_strat
import strategies.trend_join as trend_join_strat
import strategies.factor as factor_strat
import strategies.scalp as scalp_strat

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Quant Trading App — S&P 500 · NASDAQ · Crypto")

# ---------------------------------------------------------------------------
# URL query-param defaults (bookmarkable / shareable backtest links)
# Keys: ticker, strategy, start, end, capital, regime
# ---------------------------------------------------------------------------
_qp = st.query_params

def _qp_int(key: str, default: int) -> int:
    try:
        return int(_qp.get(key, default))
    except (ValueError, TypeError):
        return default

def _qp_float(key: str, default: float) -> float:
    try:
        return float(_qp.get(key, default))
    except (ValueError, TypeError):
        return default

def _qp_str(key: str, default: str) -> str:
    return _qp.get(key, default)

_PRESET_KEYS = list(MARKET_PRESETS.keys())
_STRATEGY_OPTS = [
    "Donchian Breakout",
    "VWAP + EMA + RSI Scalp",
    "Trend-Join Momentum",
    "Factor (Momentum)",
    "Regime Only",
]
_REGIME_OPTS = ["SMA Trend Filter", "3-State Markov", "Hidden Markov (HMM)"]

_default_preset_idx = _PRESET_KEYS.index(_qp_str("preset", _PRESET_KEYS[0])) \
    if _qp_str("preset", "") in _PRESET_KEYS else 0
_default_strategy_idx = _STRATEGY_OPTS.index(_qp_str("strategy", _STRATEGY_OPTS[0])) \
    if _qp_str("strategy", "") in _STRATEGY_OPTS else 0
_default_regime_idx = _REGIME_OPTS.index(_qp_str("regime", _REGIME_OPTS[0])) \
    if _qp_str("regime", "") in _REGIME_OPTS else 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Market & Instrument")

    market_preset = st.selectbox("Market Preset", _PRESET_KEYS, index=_default_preset_idx)
    preset_ticker, preset_market = MARKET_PRESETS[market_preset]

    if preset_ticker is None:
        ticker = st.text_input("Custom Ticker", value="QQQ", placeholder="e.g. QQQ, BTC-USD, NVDA")
        market_type = st.radio("Asset Class", ["equity", "crypto"], index=0)
    else:
        ticker = preset_ticker
        market_type = preset_market
        st.caption(f"Ticker: **{ticker}** · Type: **{market_type}**")

    # Crypto sub-selector
    if market_type == "crypto":
        crypto_choice = st.selectbox(
            "Crypto Asset",
            list(CRYPTO_TICKERS.keys()),
            index=list(CRYPTO_TICKERS.values()).index(ticker) if ticker in CRYPTO_TICKERS.values() else 0,
        )
        ticker = CRYPTO_TICKERS[crypto_choice]

    strategy_name = st.selectbox(
        "Strategy Family",
        _STRATEGY_OPTS,
        index=_default_strategy_idx,
    )

    _start_default = pd.Timestamp(_qp_str("start", "2019-01-01"))
    _end_default = pd.Timestamp(_qp_str("end", str(pd.Timestamp.today().date())))
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=_start_default)
    with col2:
        end_date = st.date_input("End Date", value=_end_default)

    # Timeframe — crypto can run 24/7 on any bar
    if market_type == "crypto":
        timeframe_options = ["Daily (1d)", "1-Hour (1h)", "4-Hour (4h)"]
    else:
        timeframe_options = ["Daily (1d)", "15-min (15m)", "5-min (5m)"]
    timeframe = st.selectbox("Timeframe", timeframe_options)
    tf_map = {
        "Daily (1d)": "1d", "15-min (15m)": "15m", "5-min (5m)": "5m",
        "1-Hour (1h)": "1h", "4-Hour (4h)": "4h",
    }
    interval = tf_map[timeframe]

    st.divider()
    st.subheader("Capital & Risk")
    starting_capital = st.number_input("Starting Capital ($)", value=_qp_int("capital", 100_000), step=5_000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    allow_short = st.checkbox("Allow Short Positions", value=False)

    st.divider()
    st.subheader("Regime Model")
    regime_model = st.radio("Model", _REGIME_OPTS, index=_default_regime_idx)
    regime_key = {"SMA Trend Filter": "sma", "3-State Markov": "markov", "Hidden Markov (HMM)": "hmm"}[regime_model]
    sma_period = st.slider("SMA Period", 50, 300, 200, 10)
    markov_window = st.slider("Markov Window (days)", 10, 60, 20)
    markov_threshold = st.slider("Bull/Bear Threshold (%)", 1, 15, 5) / 100

    st.divider()
    st.subheader("Strategy Parameters")
    atr_period = st.slider("ATR Period", 5, 30, 14)
    donchian_period = st.slider("Donchian Period", 5, 60, 20)
    atr_k = st.slider("ATR Multiplier (k)", 0.5, 4.0, 2.0, 0.5)
    use_weekly_confirm = st.checkbox("Weekly Donchian Confirmation", value=True)

    # Scalp-specific
    rsi_oversold = st.slider("RSI Oversold (<)", 10, 45, 30)
    rsi_overbought = st.slider("RSI Overbought (>)", 55, 90, 70)
    vwap_dist = st.slider("Max VWAP Distance (%)", 0.5, 5.0, 1.5, 0.5)

    st.divider()
    st.subheader("Costs")
    commission_bps = st.slider("Commission (bps)", 0, 20, 0)
    slippage_bps = st.slider("Slippage (bps)", 0, 50, 10)
    commission_pct = commission_bps / 10_000
    slippage_pct = slippage_bps / 10_000

    st.divider()
    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(
    tab_results, tab_compare, tab_scan, tab_regime, tab_validation, tab_audit,
    tab_capital, tab_journal, tab_firm, tab_about
) = st.tabs([
    "📊 Results", "⚖️ Compare", "🔭 Universe Scan",
    "🗺️ Regime", "🔬 Validation", "🧪 Strategy Audit",
    "💰 Capital Allocator", "📓 Journal", "🤖 Agent Firm", "ℹ️ About",
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_equity(equity: pd.Series, bah: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy",
                             line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=bah.index, y=bah, name="Buy & Hold",
                             line=dict(color="#95a5a6", width=1.5, dash="dash")))
    fig.update_layout(
        title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio ($)",
        legend=dict(orientation="h"), height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown * 100, fill="tozeroy",
        line=dict(color="#e74c3c"), fillcolor="rgba(231,76,60,0.25)", name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=260, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def kpi_tiles(metrics: dict):
    cols = st.columns(6)
    kpis = [
        ("CAGR", f"{metrics.get('cagr', 0)*100:.1f}%"),
        ("Sharpe", f"{metrics.get('sharpe', 0):.2f}"),
        ("Sortino", f"{metrics.get('sortino', 0):.2f}"),
        ("Max DD", f"{metrics.get('max_drawdown', 0)*100:.1f}%"),
        ("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%"),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
    ]
    for col, (label, val) in zip(cols, kpis):
        col.metric(label, val)


def pass_badge(passed: bool | None) -> str:
    if passed is True:
        return "✅ PASS"
    if passed is False:
        return "❌ FAIL"
    return "⚪ N/A"


# ---------------------------------------------------------------------------
# Strategy function dispatcher (for walk-forward)
# ---------------------------------------------------------------------------

def _make_strategy_fn(strategy_name, donchian_period, atr_period, atr_k,
                      sma_period, rsi_oversold, rsi_overbought, vwap_dist, allow_short):
    def strategy_fn(df_slice, **kwargs):
        if strategy_name == "Donchian Breakout":
            return donchian_strat.generate_signals(
                df_slice, period=donchian_period, atr_period=atr_period,
                atr_k=atr_k, use_weekly_confirm=False,
            )["signal"]
        elif strategy_name == "VWAP + EMA + RSI Scalp":
            return scalp_strat.generate_signals(
                df_slice, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
                max_vwap_dist_pct=vwap_dist, allow_short=allow_short,
            )["signal"]
        elif strategy_name == "Trend-Join Momentum":
            return trend_join_strat.generate_signals(
                df_slice, sma_period=sma_period, atr_period=atr_period, atr_k=atr_k,
            )["signal"]
        elif strategy_name == "Factor (Momentum)":
            return factor_strat.spy_factor_signal(df_slice)
        else:
            sma = df_slice["Close"].rolling(sma_period).mean()
            return (df_slice["Close"] > sma).astype(float).shift(1).fillna(0)
    return strategy_fn


# ============================================================================
# RUN BACKTEST
# ============================================================================

result = None
df = None
daily_df = None
regime_signal = None

if run_btn:
    start_str = str(start_date)
    end_str = str(end_date)

    # ---- Fetch data --------------------------------------------------------
    with st.spinner(f"Fetching {ticker} ({interval}) data…"):
        try:
            df = fetch(ticker, start_str, end_str, interval)
            daily_df = (
                fetch_daily_for_regime(ticker, start_str, end_str)
                if interval != "1d" else df
            )
        except DataError as e:
            st.error(f"**Data error:** {e}")
            st.stop()

    if df.empty or len(df) < 50:
        st.warning("Not enough data. Widen the date range or switch to Daily timeframe.")
        st.stop()

    # ---- Regime signal -----------------------------------------------------
    with st.spinner("Computing regime…"):
        try:
            regime_signal = get_regime_signal(
                daily_df, model=regime_key, sma_period=sma_period,
                fwd_period=markov_window, threshold=markov_threshold,
            )
        except Exception as e:
            st.warning(f"Regime model error ({e}); falling back to SMA.")
            regime_signal = get_regime_signal(daily_df, model="sma", sma_period=sma_period)

    # ---- Strategy signals --------------------------------------------------
    with st.spinner("Generating signals…"):
        try:
            if strategy_name == "Donchian Breakout":
                sig_df = donchian_strat.generate_signals(
                    df, period=donchian_period, atr_period=atr_period, atr_k=atr_k,
                    regime_signal=regime_signal,
                    use_weekly_confirm=use_weekly_confirm and interval == "1d",
                )
                signal = sig_df["signal"]
            elif strategy_name == "VWAP + EMA + RSI Scalp":
                sig_df = scalp_strat.generate_signals(
                    df, atr_period=atr_period, rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought, max_vwap_dist_pct=vwap_dist,
                    regime_signal=regime_signal, allow_short=allow_short,
                )
                signal = sig_df["signal"]
            elif strategy_name == "Trend-Join Momentum":
                sig_df = trend_join_strat.generate_signals(
                    df, sma_period=sma_period, atr_period=atr_period, atr_k=atr_k,
                    regime_signal=regime_signal,
                )
                signal = sig_df["signal"]
            elif strategy_name == "Factor (Momentum)":
                signal = factor_strat.spy_factor_signal(df, regime_signal=regime_signal)
            else:
                signal = (
                    regime_signal.reindex(df.index, method="ffill").fillna(0) > 0
                ).astype(float)
        except Exception as e:
            st.error(f"Signal generation failed: {e}")
            st.stop()

    if signal.sum() == 0:
        st.warning("No signals generated. Relax parameters, extend date range, or disable the regime gate.")
        st.stop()

    # ---- Backtest ----------------------------------------------------------
    with st.spinner("Running backtest…"):
        bt = Backtest(df, signal, {}, starting_capital, commission_pct, slippage_pct)
        result = bt.run()
        bah = buy_and_hold(df, starting_capital)

    # Persist state as URL query params so the backtest is bookmarkable/shareable
    st.query_params.update({
        "preset": market_preset,
        "strategy": strategy_name,
        "start": str(start_date),
        "end": str(end_date),
        "capital": str(starting_capital),
        "regime": regime_model,
    })

    # ========================================================================
    # TAB 1 — Results
    # ========================================================================
    with tab_results:
        st.subheader(f"Performance — {ticker} · {strategy_name}")
        badge_color = "normal" if result.metrics.get("sharpe", 0) > 1.0 else "inverse"
        kpi_tiles(result.metrics)

        extra = st.columns(4)
        extra[0].metric("Total Return", f"{result.metrics.get('total_return', 0)*100:.1f}%")
        extra[1].metric("# Trades", str(result.metrics.get("n_trades", 0)))
        extra[2].metric("Exposure", f"{result.metrics.get('exposure', 0)*100:.1f}%")
        extra[3].metric("Avg R", f"{result.metrics.get('avg_r', 0):.2f}")

        st.plotly_chart(plot_equity(result.equity, bah), use_container_width=True)
        st.plotly_chart(plot_drawdown_chart(result.drawdown), use_container_width=True)

        # Indicator overlay for scalp strategy
        if strategy_name == "VWAP + EMA + RSI Scalp" and "vwap" in sig_df.columns:
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(width=1.5)))
            fig_ind.add_trace(go.Scatter(x=df.index, y=sig_df["ema8"], name="EMA(8)", line=dict(dash="dash")))
            fig_ind.add_trace(go.Scatter(x=df.index, y=sig_df["vwap"], name="VWAP", line=dict(dash="dot")))
            fig_ind.update_layout(title="Price · VWAP · EMA(8)", height=300,
                                  margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_ind, use_container_width=True)

        # Rolling Sharpe chart
        st.divider()
        rs_window = st.slider("Rolling Sharpe Window (days)", 30, 252, 63, key="rs_window")
        roll_sh = rolling_sharpe(result.returns, window=rs_window)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=roll_sh.index, y=roll_sh, name=f"Rolling Sharpe ({rs_window}d)",
                                    line=dict(color="#f39c12", width=1.5)))
        fig_rs.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Sharpe = 1.0")
        fig_rs.add_hline(y=0.0, line_dash="dot", line_color="gray")
        fig_rs.update_layout(title="Rolling Sharpe Ratio", height=260,
                             margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_rs, use_container_width=True)

        # Benchmark metrics vs SPY
        st.divider()
        st.subheader("Benchmark Metrics vs SPY")
        with st.spinner("Computing benchmark metrics…"):
            try:
                spy_ret = fetch("SPY", str(start_date), str(end_date), "1d")["Close"].pct_change().dropna()
                bm = benchmark_metrics(result.returns, spy_ret)
                if "error" not in bm:
                    bm_cols = st.columns(5)
                    bm_cols[0].metric("Alpha (ann.)", f"{bm['alpha']*100:.1f}%")
                    bm_cols[1].metric("Beta", f"{bm['beta']:.2f}")
                    bm_cols[2].metric("Correlation", f"{bm['correlation']:.2f}")
                    bm_cols[3].metric("Info Ratio", f"{bm['information_ratio']:.2f}")
                    bm_cols[4].metric("Tracking Error", f"{bm['tracking_error']*100:.1f}%")
                else:
                    st.info(bm["error"])
            except Exception:
                st.info("Benchmark metrics unavailable (SPY fetch failed).")
                bm = None

        # Trade Log
        st.divider()
        st.subheader("Trade Log")
        if not result.trades.empty:
            st.dataframe(result.trades, use_container_width=True, height=280)
            dl_cols = st.columns(2)
            dl_cols[0].download_button(
                "⬇️ Download Trades CSV",
                result.trades.to_csv(index=False).encode(),
                f"{ticker}_trades.csv", "text/csv",
            )
            # HTML report export
            with dl_cols[1]:
                if st.button("📄 Export HTML Report"):
                    try:
                        bm_for_report = bm if isinstance(bm, dict) and "error" not in (bm or {}) else None
                        html = generate_html_report(
                            result, bah, ticker, strategy_name,
                            str(start_date), str(end_date),
                            benchmark_metrics=bm_for_report,
                        )
                        st.download_button(
                            "⬇️ Download Report HTML",
                            html.encode(),
                            f"{ticker}_{strategy_name.replace(' ', '_')}_report.html",
                            "text/html",
                            key="report_dl",
                        )
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
        else:
            st.info("No completed trades.")

    # ========================================================================
    # TAB 2 — Compare
    # ========================================================================
    with tab_compare:
        st.subheader("Strategy Comparison")
        st.caption("Run all 5 strategy families on the same ticker and date range.")
        if result is None:
            st.info("Run a backtest first (click 'Run Backtest' in the sidebar).")
        else:
            if st.button("▶ Compare All Strategies", key="compare_btn"):
                strategy_fns = {
                    "Donchian Breakout": lambda df: donchian_strat.generate_signals(
                        df, period=donchian_period, atr_period=atr_period, atr_k=atr_k,
                        use_weekly_confirm=False, regime_signal=regime_signal,
                    )["signal"],
                    "VWAP+EMA+RSI Scalp": lambda df: scalp_strat.generate_signals(
                        df, atr_period=atr_period, rsi_oversold=rsi_oversold,
                        rsi_overbought=rsi_overbought, max_vwap_dist_pct=vwap_dist,
                        regime_signal=regime_signal, allow_short=allow_short,
                    )["signal"],
                    "Trend-Join": lambda df: trend_join_strat.generate_signals(
                        df, sma_period=sma_period, atr_period=atr_period,
                        regime_signal=regime_signal,
                    )["signal"],
                    "Factor": lambda df: factor_strat.spy_factor_signal(df, regime_signal=regime_signal),
                    "Regime Only": lambda df: (
                        regime_signal.reindex(df.index, method="ffill").fillna(0) > 0
                    ).astype(float),
                }
                rows = []
                fig_cmp = go.Figure()
                bah_cmp = buy_and_hold(df, starting_capital)
                fig_cmp.add_trace(go.Scatter(x=bah_cmp.index, y=bah_cmp, name="Buy & Hold",
                                             line=dict(dash="dash", color="#95a5a6")))

                prog = st.progress(0)
                for idx, (sname, sfn) in enumerate(strategy_fns.items()):
                    with st.spinner(f"Running {sname}…"):
                        try:
                            sig = sfn(df)
                            if sig.abs().sum() == 0:
                                rows.append({"Strategy": sname, "note": "no signals"})
                                continue
                            bt_cmp = Backtest(df, sig, {}, starting_capital, commission_pct, slippage_pct)
                            r_cmp = bt_cmp.run()
                            m = r_cmp.metrics
                            rows.append({
                                "Strategy": sname,
                                "CAGR %": f"{m['cagr']*100:.1f}",
                                "Sharpe": m["sharpe"],
                                "Max DD %": f"{m['max_drawdown']*100:.1f}",
                                "Win Rate %": f"{m['win_rate']*100:.1f}",
                                "Profit Factor": m["profit_factor"],
                                "# Trades": m["n_trades"],
                            })
                            fig_cmp.add_trace(go.Scatter(x=r_cmp.equity.index, y=r_cmp.equity, name=sname))
                        except Exception as e:
                            rows.append({"Strategy": sname, "note": str(e)})
                    prog.progress((idx + 1) / len(strategy_fns))

                cmp_df = pd.DataFrame(rows)
                st.subheader("Metrics Comparison")
                if not cmp_df.empty:
                    # Highlight best Sharpe
                    st.dataframe(cmp_df, use_container_width=True)
                fig_cmp.update_layout(title=f"{ticker} — All Strategies", height=400,
                                      margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig_cmp, use_container_width=True)

    # ========================================================================
    # TAB 3 — Universe Scan
    # ========================================================================
    with tab_scan:
        st.subheader("Universe Scanner")
        st.caption("Rank tickers from the selected market universe by Sharpe ratio.")

        scan_universe_choice = st.radio(
            "Universe", ["NASDAQ 100 (top 20)", "S&P 500 (top 20)", "Crypto (all 10)"],
            horizontal=True, key="scan_universe",
        )
        scan_max = st.slider("Max Tickers", 5, 50, 20, key="scan_max")

        from engine.data import NASDAQ_100, CRYPTO_TICKERS, SP500_TICKERS
        if scan_universe_choice.startswith("NASDAQ"):
            scan_tickers = NASDAQ_100[:scan_max]
        elif scan_universe_choice.startswith("Crypto"):
            scan_tickers = list(CRYPTO_TICKERS.values())[:scan_max]
        else:
            scan_tickers = SP500_TICKERS[:scan_max]

        scan_strategy = st.selectbox("Strategy for Scan", [
            "Donchian Breakout", "VWAP+EMA+RSI Scalp", "Trend-Join",
        ], key="scan_strategy")

        if st.button("🔭 Run Universe Scan", key="scan_btn"):
            def scan_strat_fn(df_s, **kw):
                if scan_strategy == "Donchian Breakout":
                    return donchian_strat.generate_signals(df_s, period=donchian_period,
                                                           atr_period=atr_period, atr_k=atr_k,
                                                           use_weekly_confirm=False)["signal"]
                elif scan_strategy == "VWAP+EMA+RSI Scalp":
                    return scalp_strat.generate_signals(df_s, rsi_oversold=rsi_oversold,
                                                        rsi_overbought=rsi_overbought,
                                                        max_vwap_dist_pct=vwap_dist)["signal"]
                else:
                    return trend_join_strat.generate_signals(df_s, sma_period=sma_period,
                                                              atr_period=atr_period)["signal"]

            scan_progress = st.progress(0)
            scan_status = st.empty()

            def scan_cb(t, i, total):
                scan_progress.progress(i / total)
                scan_status.text(f"Scanned {i}/{total}: {t}")

            with st.spinner("Scanning universe…"):
                scan_results = scan_universe(
                    scan_tickers, scan_strat_fn, {},
                    str(start_date), str(end_date),
                    interval="1d",
                    starting_capital=starting_capital,
                    commission_pct=commission_pct,
                    slippage_pct=slippage_pct,
                    progress_callback=scan_cb,
                )

            scan_progress.progress(1.0)
            scan_status.text("Scan complete.")
            st.subheader(f"Results — {scan_strategy} on {len(scan_tickers)} tickers")
            st.dataframe(scan_results, use_container_width=True, height=400)
            if not scan_results.empty:
                st.download_button(
                    "⬇️ Download Scan Results CSV",
                    scan_results.to_csv(index=False).encode(),
                    "universe_scan.csv", "text/csv",
                )

    # ========================================================================
    # TAB — Regime
    # ========================================================================
    with tab_regime:
        st.subheader("Regime Dashboard")

        # Regime summary (markov-hedge-fund-method output contract)
        with st.spinner("Computing full regime summary…"):
            try:
                summary = regime_summary(daily_df, window=markov_window, threshold=markov_threshold)
            except Exception as e:
                summary = {"error": str(e)}

        if "error" not in summary:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Current Regime", summary.get("current_regime", "—"))
            rc2.metric("Signal Score", f"{summary.get('signal', 0):.3f}")
            rc3.metric("Regime Sharpe (WF)", f"{summary.get('backtest', {}).get('sharpe', 0):.2f}")

            col_p, col_s = st.columns(2)
            with col_p:
                st.caption("**Next-State Probabilities**")
                probs = summary.get("next_state_probabilities", {})
                st.progress(probs.get("bull", 0), text=f"Bull {probs.get('bull', 0):.1%}")
                st.progress(probs.get("sideways", 0), text=f"Sideways {probs.get('sideways', 0):.1%}")
                st.progress(probs.get("bear", 0), text=f"Bear {probs.get('bear', 0):.1%}")
            with col_s:
                st.caption("**Stationary Distribution (long-run)**")
                sd = summary.get("stationary_distribution", {})
                st.progress(sd.get("bull", 0), text=f"Bull {sd.get('bull', 0):.1%}")
                st.progress(sd.get("sideways", 0), text=f"Sideways {sd.get('sideways', 0):.1%}")
                st.progress(sd.get("bear", 0), text=f"Bear {sd.get('bear', 0):.1%}")

            # Persistence diagonal
            st.caption("**Regime Persistence (P[stay in state])**")
            pd_vals = summary.get("persistence_diagonal", {})
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Bull Persistence", f"{pd_vals.get('bull', 0):.1%}")
            pc2.metric("Sideways Persistence", f"{pd_vals.get('sideways', 0):.1%}")
            pc3.metric("Bear Persistence", f"{pd_vals.get('bear', 0):.1%}")

            # Transition matrix
            with st.expander("3×3 Transition Matrix"):
                T = summary.get("transition_matrix", [])
                if T:
                    tm_df = pd.DataFrame(T, index=["Bear", "Sideways", "Bull"],
                                         columns=["Bear", "Sideways", "Bull"])
                    st.dataframe(tm_df.style.background_gradient(cmap="RdYlGn", axis=None),
                                 use_container_width=True)
        else:
            st.warning(f"Regime summary: {summary['error']}")

        # Price + regime band chart
        rs = regime_signal.reindex(daily_df.index, method="ffill").fillna(0)
        st.plotly_chart(plot_regime(daily_df, rs, result.trades), use_container_width=True)
        st.caption("Green = Bull · Red = Bear · Gray = Sideways. Triangles = trade entries/exits.")

    # ========================================================================
    # TAB 3 — Validation
    # ========================================================================
    with tab_validation:
        st.subheader("Walk-Forward Analysis")
        strat_fn = _make_strategy_fn(
            strategy_name, donchian_period, atr_period, atr_k,
            sma_period, rsi_oversold, rsi_overbought, vwap_dist, allow_short,
        )
        with st.spinner("Running walk-forward (5 folds)…"):
            wf_df = run_walk_forward(
                df, strat_fn, {},
                n_splits=5, train_pct=0.7,
                starting_capital=starting_capital,
                commission_pct=commission_pct, slippage_pct=slippage_pct,
            )
        st.plotly_chart(plot_walk_forward(wf_df), use_container_width=True)
        mean_oos = wf_df["test_sharpe"].dropna().mean()
        oos_msg = f"Mean OOS Sharpe: **{mean_oos:.2f}**"
        if mean_oos >= 1.0:
            st.success(f"{oos_msg} ✅ (≥ 1.0 target)")
        elif mean_oos >= 0.5:
            st.warning(f"{oos_msg} ⚠️ (0.5–1.0 range — marginal)")
        else:
            st.error(f"{oos_msg} ❌ (< 0.5 — edge is weak)")
        st.dataframe(wf_df, use_container_width=True)

        st.divider()
        st.subheader("Monte Carlo Acceptance Test")
        st.caption(
            "PASS = real Sharpe ∈ [median, 95th pct] of 1,000 random P&L shuffles. "
            "FAIL_OVERFIT = suspiciously better than 95% of random paths."
        )
        pnl = result.trades["pnl_pct"] if not result.trades.empty else pd.Series(dtype=float)
        with st.spinner("Running 1,000 Monte Carlo simulations…"):
            mc = monte_carlo_test(pnl, n_sims=1_000)
        verdict = mc["verdict"]
        msg = (f"Real Sharpe {mc['real_sharpe']:.2f} · "
               f"Median sim {mc['median_sim']:.2f} · 95th pct {mc['p95_sim']:.2f}")
        if verdict == "PASS":
            st.success(f"✅ **{verdict}** — {msg}")
        elif verdict == "FAIL_OVERFIT":
            st.error(f"❌ **{verdict}** — {msg}. Strategy may be curve-fit.")
        elif verdict == "FAIL_UNDERPERFORM":
            st.warning(f"⚠️ **{verdict}** — {msg}")
        else:
            st.info(f"{verdict}")
        st.plotly_chart(plot_monte_carlo(mc), use_container_width=True)

        if strategy_name == "Donchian Breakout":
            st.divider()
            st.subheader("Parameter Sensitivity Sweep")
            with st.spinner("Running sweep…"):
                sweep = donchian_strat.parameter_sweep(df, regime_signal=regime_signal)
            if not sweep.empty:
                pivot = sweep.pivot(index="period", columns="k", values="sharpe")
                st.dataframe(
                    pivot.style.background_gradient(cmap="RdYlGn", axis=None),
                    use_container_width=True,
                )

    # ========================================================================
    # TAB 4 — Strategy Audit (6-test framework)
    # ========================================================================
    with tab_audit:
        st.subheader("6-Test Strategy Audit")
        st.caption(
            "From the *strategy-audit* skill (jackson-video-resources/skills). "
            "All 6 tests must pass for a STRONG rating."
        )
        with st.spinner("Running 6-test audit (this may take 30–60 s)…"):
            audit = strategy_audit(
                df, signal, strat_fn, {},
                starting_capital=starting_capital,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
            )

        overall = audit.get("overall", {})
        verdict = overall.get("verdict", "—")
        passed = overall.get("tests_passed", 0)
        total = overall.get("tests_total", 6)

        badge = "🟢" if verdict == "STRONG" else "🟡" if verdict == "MARGINAL" else "🔴"
        st.metric("Overall Verdict", f"{badge} {verdict}", f"{passed}/{total} tests passed")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            # Test 1
            is_ = audit.get("in_sample", {})
            st.markdown(f"**Test 1 — In-Sample** {pass_badge(is_.get('pass'))}")
            st.caption(f"Sharpe {is_.get('sharpe', 0):.2f} · Max DD {is_.get('max_drawdown', 0)*100:.1f}% · {is_.get('n_trades', 0)} trades")

            # Test 2
            wf_ = audit.get("walk_forward", {})
            st.markdown(f"**Test 2 — Walk-Forward OOS** {pass_badge(wf_.get('pass'))}")
            st.caption(f"Mean OOS Sharpe: {wf_.get('mean_oos_sharpe', 'N/A')}")

            # Test 3
            mc_ = audit.get("monte_carlo", {})
            st.markdown(f"**Test 3 — Monte Carlo** {pass_badge(mc_.get('pass'))}")
            st.caption(f"Verdict: {mc_.get('verdict', '—')} · Real Sharpe {mc_.get('real_sharpe', 0):.2f}")

        with col_b:
            # Test 4
            ps_ = audit.get("parameter_sensitivity", {})
            st.markdown(f"**Test 4 — Parameter Sensitivity** {pass_badge(ps_.get('pass'))}")
            st.caption(f"Max Sharpe drop: {ps_.get('max_sharpe_drop', 0)*100:.1f}% (threshold: 30%)")

            # Test 5
            cs_ = audit.get("cost_stress", {})
            st.markdown(f"**Test 5 — Cost Stress (2×/5×)** {pass_badge(cs_.get('pass'))}")
            for r in cs_.get("results", []):
                st.caption(f"  {r['multiplier']} costs → Sharpe {r['sharpe']:.2f}, Max DD {r['max_dd']:.1f}%")

            # Test 6
            da_ = audit.get("drawdown_analysis", {})
            st.markdown(f"**Test 6 — Drawdown Analysis** {pass_badge(da_.get('pass'))}")
            st.caption(
                f"Max DD {da_.get('max_drawdown_pct', 0):.1f}% · "
                f"Avg recovery {da_.get('avg_recovery_bars', 0):.0f} bars · "
                f"{da_.get('n_drawdown_periods', 0)} periods"
            )

# ========================================================================
# TAB 5 — Capital Allocator (always visible)
# ========================================================================
with tab_capital:
    st.subheader("Capital Allocator")
    st.caption("From the *capital-allocator* skill (jackson-video-resources/skills).")

    alloc_mode = st.radio("Mode", ["Kelly Criterion", "Multi-Strategy Weights", "Markowitz (Multi-Asset)"],
                          horizontal=True)

    if alloc_mode == "Kelly Criterion":
        st.subheader("Kelly Criterion Calculator")
        c1, c2, c3 = st.columns(3)
        wr = c1.slider("Win Rate (%)", 10, 90, 55) / 100
        aw = c2.number_input("Avg Win (%)", value=2.0, step=0.1)
        al = c3.number_input("Avg Loss (%)", value=1.0, step=0.1)
        kf = kelly_fraction(wr, aw, al)
        hk = kf * 0.5
        st.metric("Full Kelly f*", f"{kf*100:.2f}%")
        st.metric("Half Kelly (recommended)", f"{hk*100:.2f}%")
        capital_input = st.number_input("Capital ($)", value=float(starting_capital), step=1000.0)
        st.metric("Half-Kelly trade size ($)", f"${capital_input * hk:,.0f}")
        st.info("Half Kelly is recommended in practice — full Kelly is theoretically optimal but extremely volatile.")

    elif alloc_mode == "Multi-Strategy Weights":
        st.subheader("Multi-Strategy Allocation")
        n_strats = st.slider("Number of Strategies", 2, 6, 3)
        strats_input = []
        for i in range(n_strats):
            with st.expander(f"Strategy {i+1}", expanded=i == 0):
                cols = st.columns(4)
                name = cols[0].text_input("Name", value=f"Strategy {i+1}", key=f"s_name_{i}")
                sharpe = cols[1].number_input("Sharpe", value=1.2 - i * 0.2, step=0.1, key=f"s_sh_{i}")
                wr_s = cols[2].slider("Win Rate (%)", 30, 80, 55, key=f"s_wr_{i}") / 100
                dd_s = cols[3].slider("Max DD (%)", 5, 40, 15, key=f"s_dd_{i}") / 100
                strats_input.append({"name": name, "sharpe": sharpe, "win_rate": wr_s,
                                      "avg_win": 2.0, "avg_loss": 1.0, "max_dd": dd_s})
        cap_total = st.number_input("Total Capital ($)", value=float(starting_capital), step=5000.0)
        report = allocation_report(strats_input, cap_total)
        st.dataframe(report, use_container_width=True)

    else:
        st.subheader("Markowitz Portfolio Optimizer")
        st.caption("Select 2–5 assets to compute the efficient frontier and max-Sharpe portfolio.")
        all_options = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "SOL-USD", "GLD", "TLT", "AAPL", "NVDA", "MSFT"]
        selected = st.multiselect("Assets", all_options, default=["SPY", "QQQ", "BTC-USD"])
        mz_start = st.date_input("Start", value=pd.Timestamp("2021-01-01"), key="mz_start")
        mz_end = st.date_input("End", value=pd.Timestamp.today(), key="mz_end")
        if st.button("Optimize Portfolio") and len(selected) >= 2:
            with st.spinner("Fetching data and optimizing…"):
                price_data = {}
                for t in selected:
                    try:
                        price_data[t] = fetch(t, str(mz_start), str(mz_end), "1d")["Close"]
                    except DataError as e:
                        st.warning(f"Could not fetch {t}: {e}")
                if len(price_data) >= 2:
                    prices_df = pd.DataFrame(price_data).dropna()
                    returns_df = prices_df.pct_change().dropna()
                    mz = markowitz_weights(returns_df)

                    ms = mz["max_sharpe"]
                    mv = mz["min_vol"]
                    mc2, mc3 = st.columns(2)
                    with mc2:
                        st.subheader("Max-Sharpe Portfolio")
                        st.metric("Sharpe", str(ms["sharpe"]))
                        st.metric("Return", f"{ms['return']*100:.1f}%")
                        st.metric("Volatility", f"{ms['vol']*100:.1f}%")
                        st.dataframe(pd.Series(ms["weights"], name="Weight").to_frame())
                    with mc3:
                        st.subheader("Min-Volatility Portfolio")
                        st.metric("Sharpe", str(mv["sharpe"]))
                        st.metric("Return", f"{mv['return']*100:.1f}%")
                        st.metric("Volatility", f"{mv['vol']*100:.1f}%")
                        st.dataframe(pd.Series(mv["weights"], name="Weight").to_frame())

                    # Efficient frontier scatter
                    frontier = mz["frontier_df"]
                    fig_ef = go.Figure(go.Scatter(
                        x=frontier["vol"]*100, y=frontier["return"]*100,
                        mode="markers", marker=dict(color=frontier["sharpe"], colorscale="Viridis",
                                                    showscale=True, colorbar=dict(title="Sharpe")),
                    ))
                    fig_ef.update_layout(title="Efficient Frontier", xaxis_title="Volatility (%)",
                                         yaxis_title="Return (%)", height=400,
                                         margin=dict(l=40, r=20, t=40, b=40))
                    st.plotly_chart(fig_ef, use_container_width=True)

                    # ---- Correlation heatmap ----
                    st.subheader("Correlation Heatmap")
                    corr = returns_df.corr()
                    labels = list(corr.columns)
                    z = corr.values.round(2)
                    text = [[f"{v:.2f}" for v in row] for row in z]
                    fig_corr = go.Figure(go.Heatmap(
                        z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
                        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                        colorbar=dict(title="Corr"),
                    ))
                    fig_corr.update_layout(
                        title="Pairwise Return Correlations",
                        height=max(300, len(labels) * 60),
                        margin=dict(l=80, r=20, t=40, b=80),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption(
                        "Values near +1 mean assets move together; near −1 means inverse. "
                        "Diversification benefit requires low or negative cross-correlation."
                    )

                    # ---- Strategy vs universe correlation ----
                    if result is not None:
                        st.subheader("Strategy Return Correlations vs Asset Universe")
                        strat_assets = {"Strategy": result.returns}
                        for t in selected:
                            try:
                                strat_assets[t] = fetch(t, str(mz_start), str(mz_end), "1d")["Close"].pct_change().dropna()
                            except Exception:
                                pass
                        combined = pd.DataFrame(strat_assets).dropna()
                        if len(combined.columns) >= 2:
                            corr2 = combined.corr()
                            lbl2 = list(corr2.columns)
                            z2 = corr2.values.round(2)
                            txt2 = [[f"{v:.2f}" for v in row] for row in z2]
                            fig_corr2 = go.Figure(go.Heatmap(
                                z=z2, x=lbl2, y=lbl2, text=txt2, texttemplate="%{text}",
                                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                                colorbar=dict(title="Corr"),
                            ))
                            fig_corr2.update_layout(
                                title="Strategy vs Asset Returns",
                                height=max(300, len(lbl2) * 60),
                                margin=dict(l=80, r=20, t=40, b=80),
                            )
                            st.plotly_chart(fig_corr2, use_container_width=True)

# ========================================================================
# TAB 6 — Trade Journal
# ========================================================================
with tab_journal:
    st.subheader("Trade Journal")
    st.caption("From the *trade-journal* skill (jackson-video-resources/skills). Persists to ~/.quant_app_cache/trade_journal.csv")

    j_tabs = st.tabs(["Log a Trade", "Performance Review", "Journal History"])

    with j_tabs[0]:
        st.subheader("Log a Trade")
        jc = st.columns(3)
        j_ticker = jc[0].text_input("Ticker", value=ticker if ticker else "SPY", key="j_ticker")
        j_market = jc[1].selectbox("Market", ["equity", "crypto", "futures"], key="j_market")
        j_strat = jc[2].text_input("Strategy", value=strategy_name, key="j_strat")
        jc2 = st.columns(4)
        j_dir = jc2[0].radio("Direction", ["long", "short"], key="j_dir", horizontal=True)
        j_entry = jc2[1].number_input("Entry Price", value=100.0, step=0.01, key="j_entry")
        j_exit = jc2[2].number_input("Exit Price", value=102.0, step=0.01, key="j_exit")
        j_qty = jc2[3].number_input("Quantity", value=100.0, step=1.0, key="j_qty")
        jc3 = st.columns(3)
        j_stop = jc3[0].number_input("Stop Price", value=99.0, step=0.01, key="j_stop")
        j_target = jc3[1].number_input("Target Price", value=104.0, step=0.01, key="j_target")
        j_regime = jc3[2].selectbox("Regime at Entry", ["Bull", "Sideways", "Bear", "Unknown"], key="j_regime")
        j_mode = st.radio("Mode", ["paper", "live"], key="j_mode", horizontal=True)
        j_notes = st.text_area("Notes / Lessons Learned", key="j_notes")
        if st.button("Log Trade ✍️", key="log_trade_btn"):
            logged = log_trade(
                ticker=j_ticker, market=j_market, strategy=j_strat,
                direction=j_dir, entry_price=j_entry, exit_price=j_exit, quantity=j_qty,
                stop_price=j_stop, target_price=j_target, regime=j_regime, mode=j_mode,
                notes=j_notes,
            )
            pnl_val = logged["pnl_pct"]
            st.success(f"Trade logged. P&L: {pnl_val:+.2f}% · R-Multiple: {logged['r_multiple']}")

    with j_tabs[1]:
        st.subheader("Performance Review")
        jdf = get_journal()
        if not jdf.empty:
            rev = performance_review(jdf)
            if "error" not in rev:
                pr_cols = st.columns(4)
                pr_cols[0].metric("Win Rate", f"{rev['win_rate']*100:.1f}%")
                pr_cols[1].metric("Profit Factor", f"{rev['profit_factor']:.2f}")
                pr_cols[2].metric("Sharpe", f"{rev['sharpe']:.2f}")
                pr_cols[3].metric("Max DD", f"{rev['max_drawdown_pct']:.1f}%")
                pr2 = st.columns(3)
                pr2[0].metric("Avg R-Multiple", f"{rev['avg_r']:.2f}")
                pr2[1].metric("Max Consec. Losses", str(rev['max_consecutive_losses']))
                pr2[2].metric("Total Trades", str(rev['n_trades']))
                if rev.get("pnl_by_dow"):
                    st.subheader("P&L by Day of Week")
                    dow_df = pd.Series(rev["pnl_by_dow"], name="Avg P&L (%)").sort_index()
                    st.bar_chart(dow_df)
        else:
            st.info("No trades logged yet. Use the 'Log a Trade' tab to add your first entry.")

    with j_tabs[2]:
        st.subheader("Journal History")
        jdf = get_journal()
        if not jdf.empty:
            st.dataframe(jdf, use_container_width=True, height=400)
            st.download_button(
                "⬇️ Export Journal CSV",
                jdf.to_csv(index=False).encode(),
                "trade_journal.csv", "text/csv",
            )
        else:
            st.info("No trades logged yet.")

# ========================================================================
# TAB 7 — Agent Firm (paperclip-zero-human-trading-firm spec)
# ========================================================================
with tab_firm:
    st.subheader("🤖 Autonomous Agent Firm Spec")
    st.caption(
        "Based on jackson-video-resources/paperclip-zero-human-trading-firm. "
        "This tab documents the multi-agent trading org architecture. "
        "**No live orders are placed here** — this is a spec + setup guide."
    )

    st.markdown("""
### The 6-Agent Trading Firm

| Agent | Role | Gate |
|---|---|---|
| **CEO** | Central coordinator; routes tasks | Human approval required for live trading |
| **Research** | Nightly strategy discovery (YouTube, arXiv, TradingView) | Writes briefs to memory/ |
| **Backtest** | Historical validation; must hit Sharpe > 1.5 | Blocks live until approved |
| **Risk Manager** | Pre-execution gatekeeper; enforces risk-thresholds.json | Cannot be overridden by other agents |
| **Execution** | Places trades (paper by default) | Only fires after Risk Manager + human sign-off |
| **Cost Optimizer** | Weekly token-spend audit | Reports to CEO |

### Three Hard Rules
1. No agent can override its own risk limits (separation of concerns).
2. `risk-thresholds.json` is immutable except by the human.
3. If something feels off, **stop**. Don't lower thresholds to make a bad strategy fit.

### Risk Thresholds
""")

    firm_cols = st.columns(3)
    sharpe_min = firm_cols[0].number_input("Sharpe Minimum", value=1.5, step=0.1)
    max_dd = firm_cols[1].number_input("Max Drawdown (%)", value=15.0, step=1.0)
    paper_default = firm_cols[2].checkbox("Paper Trading Default", value=True)

    risk_thresholds = {
        "sharpe_minimum": sharpe_min,
        "max_drawdown_pct": max_dd,
        "paper_trading_default": paper_default,
        "live_trading_requires_board_approval": True,
    }
    import json
    st.code(json.dumps(risk_thresholds, indent=2), language="json")

    st.markdown("""
### Agent Communication Protocol (Paperclip REST API)
```bash
# Checkout a task
curl -X POST "$PAPERCLIP_API_URL/api/issues/$TASK_ID/checkout" \\
  -H "Authorization: Bearer $PAPERCLIP_API_KEY" \\
  -H "X-Paperclip-Run-Id: $RUN_ID"

# Update task with results
curl -X PATCH "$PAPERCLIP_API_URL/api/issues/$TASK_ID" \\
  -H "Authorization: Bearer $PAPERCLIP_API_KEY" \\
  -d '{"comment":"[result]","status":"done"}'
```

### Setup (run once)
```bash
npx paperclipai onboard          # install Paperclip locally
node scripts/hire_agents.js      # create all 6 agents with mandates
# Open http://localhost:3100 to see org chart
```

### VWAP Scalp Bot — Safety Check Log Schema
Mirrors `safety-check-log.json` from `claude-tradingview-mcp-trading`:
```json
{
  "ticker": "BTC-USD",
  "timestamp": "2026-06-23T10:30:00Z",
  "allowed": true,
  "direction": "long",
  "values": { "price": 65000, "ema8": 64800, "vwap": 64600, "rsi3": 28.4, "vwap_dist_pct": 0.62 },
  "long_conditions": {
    "price_gt_ema8_long": true,
    "price_gt_vwap_long": true,
    "rsi_oversold_long": true,
    "vwap_distance_ok": true
  }
}
```

### Paper Trading Outline *(disabled — for reference only)*
1. **Live data:** Binance public API (free, no auth) or Alpaca market data
2. **Signal generation:** Same `strategies/` modules, run on a cron schedule
3. **Safety check:** `strategies/scalp.safety_check()` → must return `allowed=True`
4. **Human checkpoint:** Signal queued; human approves via Telegram `/approve` command
5. **Execution:** Alpaca `api.submit_order(...)` (paper account)
6. **Audit:** Every decision logged to `safety_check_log.json` and `trade_journal.csv`

> ⚠️ No live orders in this app. Independent review required before real capital.
""")

# ========================================================================
# TAB 8 — About
# ========================================================================
with tab_about:
    st.markdown("""
## About This App

A fully self-contained local Streamlit backtester covering **S&P 500**, **NASDAQ 100**,
and **Crypto** (BTC, ETH, SOL, and others via Yahoo Finance / Binance public API).

### Integrated Skills & Repos

| Repo | Contribution |
|---|---|
| `claude-tradingview-mcp-trading` | VWAP + EMA(8) + RSI(3) scalping strategy; safety-check gate; trade CSV logging |
| `markov-hedge-fund-method` | Full Markov regime model: MLE transition matrix, Chapman-Kolmogorov forecast, stationary distribution, persistence diagonal, HMM option |
| `paperclip-zero-human-trading-firm` | Agent firm architecture spec; 6-agent org design; risk-thresholds.json schema; Paperclip API patterns |
| `skills` | `strategy-audit` 6-test framework; `capital-allocator` (Kelly, MVO, multi-strategy); `trade-journal` schema; `risk-manager` pre-trade gate |

### Strategy Families

| Strategy | Best For | Key Indicator |
|---|---|---|
| Donchian Breakout | Equities, daily | Price vs. rolling high/low channel |
| VWAP + EMA + RSI Scalp | Crypto, intraday | Session VWAP · EMA(8) · RSI(3) |
| Trend-Join Momentum | Equities | Multi-confirmation entry |
| Factor Momentum | Index ETFs | 12-1 month momentum vs. median |
| Regime Only | All | Pure regime gate as position signal |

### Regime Models

| Model | Method | Best For |
|---|---|---|
| SMA Trend Filter | Price > SMA(200) | Simple, interpretable |
| 3-State Markov | MLE transition matrix + Chapman-Kolmogorov | Regime persistence and probability forecasting |
| HMM | Gaussian HMM (hmmlearn) | Latent state discovery |

### Anti-Overfitting Checklist
- [x] `.shift(1)` on all signals — zero look-ahead
- [x] Walk-forward: 5 folds, 70/30
- [x] Monte Carlo: 1,000 shuffles, PASS ∈ [median, 95th pct]
- [x] 6-test Strategy Audit (in-sample, WF, MC, sensitivity, cost stress, drawdown)
- [x] Parameter sensitivity sweep (Donchian) — edge must hold across ±20% grid

### Disclaimers
- Backtested results do not guarantee future performance.
- Crypto markets are volatile and 24/7 — drawdowns can be severe.
- This tool is for research and educational purposes only. Not investment advice.
- Always use risk capital you can afford to lose.
""")
