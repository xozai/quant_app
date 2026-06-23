"""Vectorized backtest engine with realistic costs and slippage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.risk import compute_drawdown


@dataclass
class BacktestResult:
    equity: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    returns: pd.Series = field(default_factory=pd.Series)


class Backtest:
    def __init__(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        params: dict,
        starting_capital: float = 100_000,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.001,
    ):
        self.df = df
        self.signal = signal.reindex(df.index, fill_value=0)
        self.params = params
        self.capital = starting_capital
        self.commission = commission_pct
        self.slippage = slippage_pct

    def run(self) -> BacktestResult:
        df = self.df
        position = self.signal.shift(1).fillna(0)  # no look-ahead
        bar_returns = df["Close"].pct_change().fillna(0)

        # Strategy returns before costs
        strat_returns = position * bar_returns

        # Cost on each trade: deduct commission + slippage on every position change
        trade_mask = position.diff().fillna(0).abs() > 0
        cost_per_trade = self.commission + self.slippage
        cost_series = trade_mask.astype(float) * cost_per_trade

        net_returns = strat_returns - cost_series

        equity = self.capital * (1 + net_returns).cumprod()
        equity = pd.concat([pd.Series([self.capital], index=[df.index[0] - pd.Timedelta(days=1)]), equity])
        drawdown = compute_drawdown(equity)

        trades = self._extract_trades(df, position, net_returns)
        metrics = self._compute_metrics(net_returns, equity, trades)

        return BacktestResult(
            equity=equity,
            drawdown=drawdown,
            trades=trades,
            metrics=metrics,
            returns=net_returns,
        )

    def _extract_trades(
        self, df: pd.DataFrame, position: pd.Series, net_returns: pd.Series
    ) -> pd.DataFrame:
        """Extract individual trade records from position series."""
        entries = []
        exits = []
        in_trade = False
        entry_dt = None
        entry_px = None

        for ts, pos in position.items():
            was_in = in_trade
            in_trade_now = pos != 0

            if not was_in and in_trade_now:
                entry_dt = ts
                entry_px = df.loc[ts, "Close"] if ts in df.index else np.nan
                in_trade = True

            elif was_in and not in_trade_now:
                exit_dt = ts
                exit_px = df.loc[ts, "Close"] if ts in df.index else np.nan
                pnl = (exit_px - entry_px) / entry_px if entry_px else np.nan
                entries.append(entry_dt)
                exits.append(exit_dt)
                in_trade = False

        # Close any open trade at end
        if in_trade and entry_dt is not None:
            last_ts = df.index[-1]
            exit_px = df.loc[last_ts, "Close"]
            entries.append(entry_dt)
            exits.append(last_ts)

        if not entries:
            return pd.DataFrame(columns=["entry_dt", "exit_dt", "entry_px", "exit_px", "pnl_pct", "r_multiple"])

        records = []
        for e_dt, x_dt in zip(entries, exits):
            e_px = df.loc[e_dt, "Close"] if e_dt in df.index else np.nan
            x_px = df.loc[x_dt, "Close"] if x_dt in df.index else np.nan
            pnl = (x_px - e_px) / e_px if e_px and e_px != 0 else np.nan
            records.append({
                "entry_dt": e_dt,
                "exit_dt": x_dt,
                "entry_px": round(e_px, 4) if not np.isnan(e_px) else np.nan,
                "exit_px": round(x_px, 4) if not np.isnan(x_px) else np.nan,
                "pnl_pct": round(pnl * 100, 3) if not np.isnan(pnl) else np.nan,
                "r_multiple": round(pnl / 0.01, 2) if not np.isnan(pnl) else np.nan,
            })

        return pd.DataFrame(records)

    def _compute_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        trades: pd.DataFrame,
    ) -> dict:
        """Compute standard performance metrics."""
        n_years = len(returns) / 252 if len(returns) > 0 else 1
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        mu = returns.mean()
        sigma = returns.std()
        sharpe = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0

        downside = returns[returns < 0].std()
        sortino = (mu / downside * np.sqrt(252)) if downside > 0 else 0.0

        dd = compute_drawdown(equity)
        max_dd = float(dd.min())

        exposure = float((returns != 0).mean())
        n_trades = len(trades)

        if n_trades > 0 and "pnl_pct" in trades.columns:
            pnl = trades["pnl_pct"].dropna()
            win_rate = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
            avg_r = float(trades["r_multiple"].dropna().mean()) if "r_multiple" in trades.columns else 0.0
            gross_profit = pnl[pnl > 0].sum()
            gross_loss = abs(pnl[pnl < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            win_rate = 0.0
            avg_r = 0.0
            profit_factor = 0.0

        position = self.signal.shift(1).fillna(0)
        trades_per_year = (position.diff().abs() > 0).sum() / max(n_years, 0.01)
        turnover = trades_per_year / 252

        return {
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "avg_r": round(avg_r, 3),
            "profit_factor": round(profit_factor, 3),
            "exposure": round(exposure, 4),
            "n_trades": n_trades,
            "turnover": round(turnover, 4),
        }


def buy_and_hold(df: pd.DataFrame, starting_capital: float = 100_000) -> pd.Series:
    """Reference equity curve for buy-and-hold."""
    returns = df["Close"].pct_change().fillna(0)
    equity = starting_capital * (1 + returns).cumprod()
    return equity
