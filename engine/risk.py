"""Position sizing, R:R gate, and circuit breaker logic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    """Fixed-fractional sizing. Returns number of shares (always >= 1)."""
    risk_dollar = capital * risk_pct
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return 1
    shares = int(risk_dollar / stop_distance)
    return max(1, shares)


def check_rr(entry: float, stop: float, target: float, min_rr: float = 2.0) -> bool:
    """Return True if reward/risk ratio meets the minimum threshold."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return False
    return (reward / risk) >= min_rr


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Return drawdown series (0 to -1) from equity curve."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return dd


def apply_circuit_breakers(
    equity: pd.Series,
    daily_loss_limit: float = 0.025,
    max_dd_limit: float = 0.08,
) -> pd.Series:
    """
    Zero out returns once daily loss > daily_loss_limit OR drawdown > max_dd_limit.
    Returns a mask Series (True = trading allowed, False = circuit breaker tripped).
    """
    returns = equity.pct_change().fillna(0)
    allowed = pd.Series(True, index=equity.index)
    peak = equity.iloc[0]
    day_start_equity = equity.iloc[0]
    prev_date = equity.index[0].date() if hasattr(equity.index[0], "date") else None

    for i, (ts, eq) in enumerate(equity.items()):
        current_date = ts.date() if hasattr(ts, "date") else ts
        if current_date != prev_date:
            day_start_equity = eq
            prev_date = current_date

        peak = max(peak, eq)
        drawdown = (eq - peak) / peak if peak > 0 else 0
        daily_loss = (eq - day_start_equity) / day_start_equity if day_start_equity > 0 else 0

        if drawdown < -max_dd_limit or daily_loss < -daily_loss_limit:
            allowed.iloc[i:] = False
            break

    return allowed


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()
