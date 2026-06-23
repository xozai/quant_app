"""
Universe scanner — run a strategy across multiple tickers and rank by Sharpe.
Backs the 'Universe Scan' tab in app.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import fetch, DataError
from engine.backtest import Backtest


def scan_universe(
    tickers: list[str],
    strategy_fn,
    strategy_params: dict,
    start: str,
    end: str,
    interval: str = "1d",
    starting_capital: float = 100_000,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.001,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run strategy_fn on each ticker and return a ranked DataFrame.

    strategy_fn(df, **params) -> pd.Series of signals

    progress_callback(ticker, i, total) called after each ticker completes.
    Returns DataFrame sorted by Sharpe descending with columns:
      ticker, sharpe, cagr, max_drawdown, win_rate, profit_factor, n_trades, error
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            df = fetch(ticker, start, end, interval)
            if df.empty or len(df) < 60:
                results.append({"ticker": ticker, "error": "insufficient data"})
                continue

            signal = strategy_fn(df, **strategy_params)
            if signal.abs().sum() == 0:
                results.append({"ticker": ticker, "error": "no signals generated"})
                continue

            bt = Backtest(df, signal, {}, starting_capital, commission_pct, slippage_pct)
            r = bt.run()
            m = r.metrics
            results.append({
                "ticker": ticker,
                "sharpe": m.get("sharpe", np.nan),
                "cagr_%": round(m.get("cagr", 0) * 100, 2),
                "max_drawdown_%": round(m.get("max_drawdown", 0) * 100, 2),
                "win_rate_%": round(m.get("win_rate", 0) * 100, 1),
                "profit_factor": m.get("profit_factor", np.nan),
                "n_trades": m.get("n_trades", 0),
                "error": "",
            })
        except DataError as e:
            results.append({"ticker": ticker, "error": str(e)})
        except Exception as e:
            results.append({"ticker": ticker, "error": f"unexpected: {e}"})

        if progress_callback:
            progress_callback(ticker, i + 1, total)

    df_results = pd.DataFrame(results)
    if "sharpe" in df_results.columns:
        ok = df_results["error"].isna() | (df_results["error"] == "")
        df_results = pd.concat([
            df_results[ok].sort_values("sharpe", ascending=False),
            df_results[~ok],
        ]).reset_index(drop=True)

    return df_results
