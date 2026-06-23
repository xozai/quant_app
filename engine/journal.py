"""
Trade journal and performance logger.
Implements the 'trade-journal' skill from jackson-video-resources/skills.
Writes and reads a local CSV log; computes per-session analytics.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

JOURNAL_PATH = Path.home() / ".quant_app_cache" / "trade_journal.csv"
SAFETY_LOG_PATH = Path.home() / ".quant_app_cache" / "safety_check_log.json"

JOURNAL_COLS = [
    "date", "time_utc", "ticker", "market", "strategy",
    "direction", "entry_price", "exit_price", "quantity",
    "pnl_usd", "pnl_pct", "r_multiple",
    "stop_price", "target_price",
    "regime", "signal_strength",
    "mode",           # paper | live
    "notes",
]


def _load() -> pd.DataFrame:
    if JOURNAL_PATH.exists():
        return pd.read_csv(JOURNAL_PATH)
    return pd.DataFrame(columns=JOURNAL_COLS)


def _save(df: pd.DataFrame):
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(JOURNAL_PATH, index=False)


def log_trade(
    ticker: str,
    market: str,
    strategy: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    stop_price: float = np.nan,
    target_price: float = np.nan,
    regime: str = "",
    signal_strength: float = np.nan,
    mode: str = "paper",
    notes: str = "",
) -> dict:
    """Append one trade to the journal CSV. Returns the logged row as a dict."""
    now = pd.Timestamp.utcnow()
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0
    pnl_usd = pnl_pct * entry_price * quantity
    stop_dist = abs(entry_price - stop_price) if not np.isnan(stop_price) else np.nan
    r_multiple = (exit_price - entry_price) / stop_dist if (stop_dist and stop_dist > 0) else np.nan

    row = {
        "date": now.strftime("%Y-%m-%d"),
        "time_utc": now.strftime("%H:%M:%S"),
        "ticker": ticker,
        "market": market,
        "strategy": strategy,
        "direction": direction,
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "quantity": quantity,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct * 100, 3),
        "r_multiple": round(r_multiple, 2) if not np.isnan(r_multiple) else np.nan,
        "stop_price": stop_price,
        "target_price": target_price,
        "regime": regime,
        "signal_strength": signal_strength,
        "mode": mode,
        "notes": notes,
    }

    df = _load()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)
    return row


def get_journal() -> pd.DataFrame:
    return _load()


def performance_review(df: pd.DataFrame | None = None) -> dict:
    """
    Compute the full performance metrics matching the trade-journal skill output:
    returns, Sharpe, win rate, profit factor, max consecutive losses,
    max drawdown, patterns by day-of-week.
    """
    if df is None:
        df = _load()
    if df.empty:
        return {"error": "No trades logged yet."}

    df = df.copy()
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")

    pnl = df["pnl_pct"].dropna()
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = len(wins) / n if n > 0 else 0
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    mu = pnl.mean()
    sigma = pnl.std()
    sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0

    # Max consecutive losses
    signs = np.sign(pnl.values)
    max_consec_loss = 0
    cur = 0
    for s in signs:
        if s < 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    # Equity curve + max drawdown
    equity = (1 + pnl / 100).cumprod()
    peak = equity.cummax()
    dd = ((equity - peak) / peak)
    max_dd = float(dd.min())

    # Day-of-week breakdown
    if "date" in df.columns:
        df["dow"] = pd.to_datetime(df["date"], errors="coerce").dt.day_name()
        dow = df.groupby("dow")["pnl_pct"].mean().round(3).to_dict()
    else:
        dow = {}

    return {
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "avg_pnl_pct": round(float(mu), 3),
        "avg_r": round(float(df["r_multiple"].dropna().mean()), 3) if not df["r_multiple"].dropna().empty else 0,
        "max_consecutive_losses": max_consec_loss,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "gross_win_pct": round(gross_win, 2),
        "gross_loss_pct": round(gross_loss, 2),
        "pnl_by_dow": dow,
    }


def log_safety_check(ticker: str, check_result: dict):
    """Append a safety-check decision to the JSON log (mirrors bot.js safety-check-log.json)."""
    SAFETY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if SAFETY_LOG_PATH.exists():
        try:
            with open(SAFETY_LOG_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    check_result["ticker"] = ticker
    check_result["timestamp"] = pd.Timestamp.utcnow().isoformat()
    existing.append(check_result)
    with open(SAFETY_LOG_PATH, "w") as f:
        json.dump(existing[-500:], f, indent=2)  # keep last 500 decisions
