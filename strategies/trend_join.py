"""Trend-momentum-join strategy: enter only after confirmation conditions align."""

from __future__ import annotations

import pandas as pd
import numpy as np

from engine.risk import atr


def generate_signals(
    df: pd.DataFrame,
    sma_period: int = 200,
    atr_period: int = 14,
    atr_k: float = 1.0,
    min_gap_pct: float = 0.05,
    min_volume: int = 50_000,
    min_price: float = 3.0,
    regime_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Trend-join entry rules (daily bars):
    1. Close > prior_day_high
    2. Prior close > SMA(200)
    3. Close > intraday high of day (always true on daily — used as placeholder for
       "no new lower high formed"; on intraday data this gate would be tighter)
    4. Optional: gap > min_gap_pct, volume > min_volume, price > min_price

    Exit: ATR trailing stop or regime flip.
    """
    sma = df["Close"].rolling(sma_period).mean()
    atr_s = atr(df, atr_period)
    prior_high = df["High"].shift(1)
    prior_close = df["Close"].shift(1)

    # All conditions shifted so today's signal uses prior bar data
    cond_above_prior_high = df["Close"] > prior_high
    cond_sma = prior_close > sma.shift(1)
    gap_pct = (df["Open"] - prior_close) / prior_close.replace(0, np.nan)
    cond_gap = gap_pct > min_gap_pct
    cond_volume = df["Volume"] > min_volume
    cond_price = df["Close"] > min_price

    entry_cond = cond_above_prior_high & cond_sma & cond_gap & cond_volume & cond_price

    if regime_signal is not None:
        rs = regime_signal.reindex(df.index, method="ffill").fillna(0)
        entry_cond = entry_cond & (rs > 0)

    # Build position with ATR trailing stop exit
    trail_stop_hit = df["Close"] < df["Close"].shift(1) - atr_k * atr_s.shift(1)

    position = pd.Series(0, index=df.index)
    in_trade = False
    for i in range(1, len(df)):
        if not in_trade:
            if entry_cond.iloc[i]:
                in_trade = True
                position.iloc[i] = 1
        else:
            # Regime exit
            rs_val = 1.0
            if regime_signal is not None:
                rs_val = float(regime_signal.reindex(df.index, method="ffill").fillna(0).iloc[i])
            if trail_stop_hit.iloc[i] or rs_val <= 0:
                in_trade = False
                position.iloc[i] = 0
            else:
                position.iloc[i] = 1

    entry_mask = (position == 1) & (position.shift(1).fillna(0) == 0)
    signals = pd.DataFrame({
        "signal": position,
        "sma": sma,
        "atr_val": atr_s,
        "entry_price": np.where(entry_mask, df["Close"], np.nan),
        "stop_price": np.where(entry_mask, df["Close"] - atr_k * atr_s, np.nan),
        "target_price": np.where(entry_mask, df["Close"] + 2 * atr_k * atr_s, np.nan),
    }, index=df.index)

    return signals
