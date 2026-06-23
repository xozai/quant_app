"""Donchian channel breakout strategy with higher-timeframe confirmation and ATR trailing stop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.risk import atr


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Return upper and lower Donchian channel bands."""
    upper = df["High"].rolling(period).max()
    lower = df["Low"].rolling(period).min()
    return pd.DataFrame({"upper": upper, "lower": lower}, index=df.index)


def _weekly_upper(daily_df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute upper Donchian on weekly resampled data, forward-filled to daily index."""
    weekly = daily_df["High"].resample("W").max()
    weekly_upper = weekly.rolling(period).max()
    return weekly_upper.reindex(daily_df.index, method="ffill")


def generate_signals(
    df: pd.DataFrame,
    period: int = 20,
    atr_period: int = 14,
    atr_k: float = 2.0,
    regime_signal: pd.Series | None = None,
    use_weekly_confirm: bool = True,
) -> pd.DataFrame:
    """
    Generate long entry/exit signals for the Donchian breakout strategy.

    Returns a DataFrame with columns:
      signal      : 1 = long, 0 = flat  (already shifted — no look-ahead)
      entry_price : price at signal bar close (approximate)
      stop_price  : ATR-based stop below entry
      target_price: 2× ATR above entry
      atr_val     : ATR value at entry
    """
    dc = donchian_channel(df, period)
    atr_s = atr(df, atr_period)

    # Entry: close breaks above the prior bar's upper channel
    breakout = df["Close"] > dc["upper"].shift(1)

    if use_weekly_confirm:
        weekly_upper = _weekly_upper(df, period)
        weekly_confirm = df["Close"] > weekly_upper.shift(1)
        breakout = breakout & weekly_confirm

    # Regime gate
    if regime_signal is not None:
        # Reindex regime to match df (may be daily mapped to intraday)
        rs = regime_signal.reindex(df.index, method="ffill").fillna(0)
        breakout = breakout & (rs > 0)

    # Trailing stop: exit when close drops more than k * ATR below the running high
    # We track this bar-by-bar after entry; simplify to: exit when close < prior_close - k*ATR
    trail_stop_hit = df["Close"] < df["Close"].shift(1) - atr_k * atr_s.shift(1)

    # Build position series: enter on breakout, exit on trail stop
    position = pd.Series(0, index=df.index)
    in_trade = False
    for i in range(1, len(df)):
        if not in_trade:
            if breakout.iloc[i]:
                in_trade = True
                position.iloc[i] = 1
        else:
            if trail_stop_hit.iloc[i]:
                in_trade = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1

    signals = pd.DataFrame({
        "signal": position,
        "upper": dc["upper"],
        "lower": dc["lower"],
        "atr_val": atr_s,
    }, index=df.index)

    # Entry/stop/target for position sizing — filled only at entry bars
    entry_mask = (signals["signal"] == 1) & (signals["signal"].shift(1).fillna(0) == 0)
    signals["entry_price"] = np.where(entry_mask, df["Close"], np.nan)
    signals["stop_price"] = np.where(entry_mask, df["Close"] - atr_k * atr_s, np.nan)
    signals["target_price"] = np.where(entry_mask, df["Close"] + 2 * atr_k * atr_s, np.nan)

    return signals


def parameter_sweep(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    k_values: list[float] | None = None,
    regime_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Run a grid of (period, k) combinations. Returns DataFrame of results
    for use in the sensitivity sweep display.
    """
    from engine.backtest import Backtest

    periods = periods or [10, 15, 20, 30, 40]
    k_values = k_values or [1.5, 2.0, 2.5]
    rows = []
    for p in periods:
        for k in k_values:
            try:
                signals = generate_signals(df, period=p, atr_period=14, atr_k=k, regime_signal=regime_signal)
                bt = Backtest(df, signals["signal"], {})
                result = bt.run()
                rows.append({
                    "period": p, "k": k,
                    "sharpe": round(result.metrics.get("sharpe", np.nan), 3),
                    "cagr": round(result.metrics.get("cagr", np.nan), 4),
                    "max_dd": round(result.metrics.get("max_drawdown", np.nan), 4),
                })
            except Exception:
                rows.append({"period": p, "k": k, "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan})
    return pd.DataFrame(rows)
