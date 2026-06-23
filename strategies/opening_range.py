"""Opening-range scalp strategy (requires intraday data: 1-min or 5-min)."""

from __future__ import annotations

import pandas as pd
import numpy as np

from engine.risk import atr


def _session_open_range(intraday_df: pd.DataFrame, or_minutes: int = 15) -> pd.DataFrame:
    """
    For each trading session, compute the opening-range high/low/close/open
    using the first `or_minutes` of data.
    Returns a daily-indexed DataFrame with or_high, or_low, or_open, or_close.
    """
    # Determine bar frequency in minutes
    if len(intraday_df) < 2:
        return pd.DataFrame()

    freq_minutes = int((intraday_df.index[1] - intraday_df.index[0]).total_seconds() / 60)
    freq_minutes = max(1, freq_minutes)
    bars_in_or = max(1, or_minutes // freq_minutes)

    records = []
    for date, group in intraday_df.groupby(intraday_df.index.date):
        or_bars = group.iloc[:bars_in_or]
        if or_bars.empty:
            continue
        records.append({
            "date": pd.Timestamp(date),
            "or_high": or_bars["High"].max(),
            "or_low": or_bars["Low"].min(),
            "or_open": or_bars["Open"].iloc[0],
            "or_close": or_bars["Close"].iloc[-1],
        })
    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records).set_index("date")
    return result


def generate_signals(
    intraday_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    or_minutes: int = 15,
    atr_period: int = 14,
    atr_threshold: float = 0.25,
    time_stop_hour: int = 11,
    time_stop_minute: int = 30,
    regime_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Opening-range scalp on intraday data.
    Requires a daily_df for ATR reference.
    Returns intraday-indexed signal DataFrame.
    """
    or_data = _session_open_range(intraday_df, or_minutes)
    if or_data.empty:
        return pd.DataFrame({"signal": pd.Series(0, index=intraday_df.index)})

    daily_atr = atr(daily_df, atr_period)

    position = pd.Series(0, index=intraday_df.index)
    entry_price = pd.Series(np.nan, index=intraday_df.index)
    stop_price = pd.Series(np.nan, index=intraday_df.index)
    target_price = pd.Series(np.nan, index=intraday_df.index)

    in_trade = False
    trade_direction = 0
    trade_entry = np.nan
    trade_stop = np.nan
    trade_target = np.nan
    trade_date = None

    for ts, row in intraday_df.iterrows():
        date = ts.date()
        date_ts = pd.Timestamp(date)

        # Check time stop
        if in_trade:
            t = ts.time()
            if t.hour > time_stop_hour or (t.hour == time_stop_hour and t.minute >= time_stop_minute):
                in_trade = False
                trade_direction = 0
                continue
            # Check target / stop hit
            if trade_direction == 1:
                if row["Low"] <= trade_stop or row["High"] >= trade_target:
                    in_trade = False
                    trade_direction = 0
                    continue
                position[ts] = 1
            elif trade_direction == -1:
                if row["High"] >= trade_stop or row["Low"] <= trade_target:
                    in_trade = False
                    trade_direction = 0
                    continue
                position[ts] = -1
            continue

        if date_ts not in or_data.index:
            continue
        if trade_date == date:
            continue  # one trade per session

        or_row = or_data.loc[date_ts]
        or_range = or_row["or_high"] - or_row["or_low"]

        # Need the previous day's ATR
        prev_dates = daily_atr.index[daily_atr.index < date_ts]
        if len(prev_dates) == 0:
            continue
        d_atr = daily_atr.loc[prev_dates[-1]]

        if or_range < atr_threshold * d_atr:
            continue  # OR not large enough — not a manipulation candle

        # Regime gate
        if regime_signal is not None:
            prev_regime_dates = regime_signal.index[regime_signal.index <= date_ts]
            if len(prev_regime_dates) == 0:
                continue
            rs_val = float(regime_signal.loc[prev_regime_dates[-1]])
            if rs_val <= 0:
                continue

        # Direction: fade opposite to OR candle direction
        or_bullish = or_row["or_close"] > or_row["or_open"]
        # After OR completes, check if we're past the OR window
        or_end_time = (pd.Timestamp(date) + pd.Timedelta(minutes=or_minutes)).time()
        if ts.time() < or_end_time:
            continue

        # Fibonacci 38.2% retracement target
        fib_382 = or_row["or_high"] - 0.382 * or_range

        if not or_bullish:
            # Fade long: enter at or_low, stop at or_high, target at 38.2% from top
            trade_direction = 1
            trade_entry = or_row["or_low"]
            trade_stop = or_row["or_high"]
            trade_target = or_row["or_low"] + 0.382 * or_range
        else:
            # Fade short: enter at or_high, stop at or_low, target 38.2% from bottom
            trade_direction = -1
            trade_entry = or_row["or_high"]
            trade_stop = or_row["or_low"]
            trade_target = or_row["or_high"] - 0.382 * or_range

        in_trade = True
        trade_date = date
        position[ts] = trade_direction
        entry_price[ts] = trade_entry
        stop_price[ts] = trade_stop
        target_price[ts] = trade_target

    return pd.DataFrame({
        "signal": position,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
    }, index=intraday_df.index)
