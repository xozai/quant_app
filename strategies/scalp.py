"""
VWAP + EMA(8) + RSI(3) scalping strategy.
Ported from jackson-video-resources/claude-tradingview-mcp-trading (bot.js).
Works on any intraday OHLCV DataFrame (1m, 5m, 15m, 4h) and on daily crypto data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.risk import atr


# ---------------------------------------------------------------------------
# Indicator calculations (mirroring bot.js logic in Python)
# ---------------------------------------------------------------------------

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session VWAP — resets daily at UTC midnight.
    For daily data or crypto (no session break), uses a rolling 20-bar VWAP.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    dollar_vol = typical * df["Volume"]

    # Group by calendar date for intraday; rolling for daily/crypto
    if hasattr(df.index, "date"):
        cum_dv = dollar_vol.groupby(df.index.date).cumsum()
        cum_v = df["Volume"].groupby(df.index.date).cumsum()
    else:
        cum_dv = dollar_vol.rolling(20).sum()
        cum_v = df["Volume"].rolling(20).sum()

    return cum_dv / cum_v.replace(0, np.nan)


def calc_rsi(series: pd.Series, period: int = 3) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------------------------------------------------
# Safety-check gate (mirrors bot.js multi-condition engine)
# ---------------------------------------------------------------------------

def safety_check(
    row: pd.Series,
    ema8: float,
    vwap: float,
    rsi3: float,
    max_vwap_distance_pct: float = 1.5,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
) -> dict:
    """
    All conditions must pass for a trade to be allowed.
    Returns dict with 'allowed', 'direction', and per-condition results.
    Mirrors the decision logic in bot.js safety-check-log.json.
    """
    price = float(row["Close"])
    vwap_dist_pct = abs(price - vwap) / vwap * 100 if vwap else 999

    conditions = {
        "price_gt_ema8_long": price > ema8,
        "price_gt_vwap_long": price > vwap,
        "rsi_oversold_long": rsi3 < rsi_oversold,
        "vwap_distance_ok": vwap_dist_pct <= max_vwap_distance_pct,
    }
    short_conditions = {
        "price_lt_ema8_short": price < ema8,
        "price_lt_vwap_short": price < vwap,
        "rsi_overbought_short": rsi3 > rsi_overbought,
        "vwap_distance_ok": vwap_dist_pct <= max_vwap_distance_pct,
    }

    long_ok = all(conditions.values())
    short_ok = all(short_conditions.values())

    return {
        "allowed": long_ok or short_ok,
        "direction": "long" if long_ok else ("short" if short_ok else "flat"),
        "long_conditions": conditions,
        "short_conditions": short_conditions,
        "values": {"price": price, "ema8": ema8, "vwap": vwap, "rsi3": rsi3, "vwap_dist_pct": vwap_dist_pct},
    }


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    ema_period: int = 8,
    rsi_period: int = 3,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    max_vwap_dist_pct: float = 1.5,
    atr_period: int = 14,
    stop_pct: float = 0.003,
    regime_signal: pd.Series | None = None,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    VWAP + EMA(8) + RSI(3) scalping strategy.

    Entry long:  price > VWAP AND price > EMA(8) AND RSI(3) < oversold AND within VWAP distance
    Entry short: price < VWAP AND price < EMA(8) AND RSI(3) > overbought AND within VWAP distance
    Exit:        RSI(3) crosses 50, OR 0.3% stop hit

    Returns DataFrame with: signal, ema8, vwap, rsi3, entry_price, stop_price, target_price
    """
    ema8 = calc_ema(df["Close"], ema_period)
    vwap = calc_vwap(df)
    rsi3 = calc_rsi(df["Close"], rsi_period)
    atr_s = atr(df, atr_period)

    # Long entries
    long_entry = (
        (df["Close"] > vwap) &
        (df["Close"] > ema8) &
        (rsi3 < rsi_oversold) &
        ((df["Close"] - vwap).abs() / vwap * 100 <= max_vwap_dist_pct)
    )

    # Short entries
    short_entry = (
        (df["Close"] < vwap) &
        (df["Close"] < ema8) &
        (rsi3 > rsi_overbought) &
        ((df["Close"] - vwap).abs() / vwap * 100 <= max_vwap_dist_pct)
    ) if allow_short else pd.Series(False, index=df.index)

    # Regime gate
    if regime_signal is not None:
        rs = regime_signal.reindex(df.index, method="ffill").fillna(0)
        long_entry = long_entry & (rs > 0)
        short_entry = short_entry & (rs < 0)

    # RSI cross-50 exit signal
    rsi_cross_50 = (rsi3 > 50) & (rsi3.shift(1) <= 50)

    # Build position series
    position = pd.Series(0, index=df.index)
    in_trade = False
    direction = 0
    entry_px = np.nan

    for i in range(1, len(df)):
        if not in_trade:
            if long_entry.shift(1).fillna(False).iloc[i]:
                in_trade = True
                direction = 1
                entry_px = df["Close"].iloc[i]
                position.iloc[i] = 1
            elif short_entry.shift(1).fillna(False).iloc[i] and allow_short:
                in_trade = True
                direction = -1
                entry_px = df["Close"].iloc[i]
                position.iloc[i] = -1
        else:
            price = df["Close"].iloc[i]
            stop_hit = abs(price - entry_px) / entry_px > stop_pct and (
                (direction == 1 and price < entry_px) or
                (direction == -1 and price > entry_px)
            )
            rsi_exit = rsi_cross_50.iloc[i]

            if stop_hit or rsi_exit:
                in_trade = False
                direction = 0
                position.iloc[i] = 0
            else:
                position.iloc[i] = direction

    entry_mask = (position != 0) & (position.shift(1).fillna(0) == 0)

    signals = pd.DataFrame({
        "signal": position,
        "ema8": ema8,
        "vwap": vwap,
        "rsi3": rsi3,
        "atr_val": atr_s,
        "entry_price": np.where(entry_mask, df["Close"], np.nan),
        "stop_price": np.where(entry_mask, df["Close"] * (1 - stop_pct), np.nan),
        "target_price": np.where(entry_mask, df["Close"] + 2 * atr_s, np.nan),
    }, index=df.index)

    return signals
