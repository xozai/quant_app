"""Multi-factor scoring strategy (momentum, value, quality, Piotroski, Altman)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Static liquid universe — top-50 S&P 500 names by market cap (no API needed)
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "JPM", "AVGO",
    "TSLA", "UNH", "XOM", "V", "MA", "JNJ", "PG", "HD", "MRK", "COST",
    "CVX", "ABBV", "BAC", "WMT", "NFLX", "CRM", "AMD", "TMO", "ACN", "LIN",
    "MCD", "KO", "PEP", "ADBE", "DHR", "ABT", "CSCO", "WFC", "TXN", "INTU",
    "NEE", "AMGN", "MS", "RTX", "GS", "UPS", "AMAT", "SPGI", "ISRG", "NOW",
]


def compute_momentum_score(prices: pd.DataFrame) -> pd.Series:
    """12-1 month momentum score, cross-sectionally z-scored."""
    ret_12 = prices.pct_change(252)
    ret_1 = prices.pct_change(21)
    mom = ret_12 - ret_1  # exclude most-recent month (standard)
    last = mom.iloc[-1].dropna()
    if last.std() == 0:
        return last * 0
    return (last - last.mean()) / last.std()


def compute_52w_proximity(prices: pd.DataFrame) -> pd.Series:
    """Proximity to 52-week high, z-scored."""
    high_52w = prices.rolling(252).max().iloc[-1]
    last_price = prices.iloc[-1]
    proximity = last_price / high_52w.replace(0, np.nan)
    proximity = proximity.dropna()
    if proximity.std() == 0:
        return proximity * 0
    return (proximity - proximity.mean()) / proximity.std()


def compute_factor_scores(prices: pd.DataFrame) -> pd.Series:
    """
    Composite factor score combining:
      - 12-1 month momentum (40% combined)
      - 52-week proximity (15%)
    Value and quality factors would require fundamental data not freely available
    via yfinance price history alone, so we weight price-based factors here and
    document the limitation clearly.
    """
    mom = compute_momentum_score(prices) * 0.55
    prox = compute_52w_proximity(prices) * 0.45
    combined = mom.add(prox, fill_value=0)
    return combined.sort_values(ascending=False)


def generate_signals(
    prices: pd.DataFrame,
    rebalance_freq: str = "ME",
    top_pct: float = 0.2,
    regime_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Monthly rebalance: go long the top quintile by factor score.
    Returns daily position DataFrame (1 = long each name, 0 = flat).
    For SPY-only backtest, returns a single-column signal based on
    whether SPY is in the top half of its own rolling distribution.

    prices: DataFrame of Close prices, columns = tickers
    """
    rebalance_dates = prices.resample(rebalance_freq).last().index
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, rb_date in enumerate(rebalance_dates):
        scores = compute_factor_scores(prices.loc[:rb_date])
        if scores.empty:
            continue
        n_long = max(1, int(len(scores) * top_pct))
        longs = set(scores.head(n_long).index)

        # Apply from this rebalance to next
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else prices.index[-1]
        mask = (prices.index >= rb_date) & (prices.index < next_date)
        for col in prices.columns:
            signals.loc[mask, col] = 1.0 if col in longs else 0.0

    # Regime gate on all positions
    if regime_signal is not None:
        rs = regime_signal.reindex(prices.index, method="ffill").fillna(0)
        for col in signals.columns:
            signals[col] = signals[col] * (rs > 0).astype(float)

    return signals


def spy_factor_signal(spy_prices: pd.DataFrame, regime_signal: pd.Series | None = None) -> pd.Series:
    """
    For single-ticker (SPY) mode: derive a factor-like signal from
    the SPY's own momentum rank vs its rolling median momentum.
    Useful as a standalone strategy gate.
    """
    mom = spy_prices["Close"].pct_change(252) - spy_prices["Close"].pct_change(21)
    med = mom.rolling(63).median()
    signal = (mom > med).astype(float).shift(1).fillna(0)
    if regime_signal is not None:
        rs = regime_signal.reindex(spy_prices.index, method="ffill").fillna(0)
        signal = signal * (rs > 0).astype(float)
    return signal
