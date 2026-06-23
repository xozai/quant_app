# Quant Trading App — Local Streamlit Backtester

A fully self-contained, locally runnable Streamlit application for systematic
trading strategy design, backtesting, and visualization on the S&P 500 (SPY).
No paid API keys required. All data from Yahoo Finance (yfinance), cached to disk.

---

## Quick Start

```bash
cd quant_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py             # opens http://localhost:8501
```

Click **Run Backtest** with the default settings — results appear immediately.

---

## Run Tests

```bash
pytest tests/ -v
```

Five smoke tests cover data fetching, the Donchian backtest engine,
regime labeling, Monte Carlo output structure, ATR correctness, and
drawdown non-positivity.

---

## Project Structure

```
quant_app/
├── app.py                  Streamlit UI
├── strategies/
│   ├── regime.py           SMA trend filter + 3-state Markov regime model
│   ├── donchian.py         Donchian channel breakout (default)
│   ├── trend_join.py       Trend-momentum-join entry rules
│   ├── factor.py           Multi-factor scoring (price-based proxies)
│   └── opening_range.py    Opening-range scalp (intraday)
├── engine/
│   ├── data.py             yfinance fetch + parquet cache (~/.quant_app_cache/)
│   ├── backtest.py         Vectorized engine with costs & slippage
│   ├── risk.py             Position sizing, circuit breakers, R:R gate
│   └── validation.py       Walk-forward + Monte Carlo acceptance test
├── tests/
│   └── test_smoke.py
└── requirements.txt
```

---

## Strategy Spec

### 1. Donchian Channel Breakout (default)

**Thesis**: Price breaking above its recent high (Donchian upper band) signals
trend initiation. Confirmed on the weekly timeframe to filter noise. A trailing
ATR stop captures the trend and controls downside.

**Entry**: Close > prior bar's Donchian upper band (rolling max of High over N bars).
Optional: same breakout required on weekly resampled data.

**Exit**: ATR trailing stop — exit when close < prior close − k × ATR(14).

**Regime gate**: Only trade long when price > SMA(200) (or Markov score > 0).

**Parameters**:
- `period`: Donchian lookback (default 20)
- `atr_period`: ATR smoothing period (default 14)
- `k`: ATR stop multiplier (default 2.0)

**Known failure modes**:
- Choppy, range-bound markets produce repeated whipsaws
- Weekly confirmation may filter valid signals in fast-moving markets
- ATR trailing stop can be too tight during high-volatility events

---

### 2. Trend-Join Momentum

**Thesis**: Enter a trend only after multiple confirmations align — breakout above
prior-day high, price above the 200-SMA, gap-up with volume. This reduces chasing
and focuses on high-conviction continuation setups.

**Entry**: All of: (a) Close > prior High; (b) prior Close > SMA(200);
(c) gap > 5%; (d) Volume > 50k; (e) Price > $3.

**Exit**: ATR trailing stop; or if regime score flips negative.

---

### 3. Factor (SPY Momentum)

**Thesis**: SPY's own 12-1 month momentum relative to its rolling median predicts
near-term direction — a simplified factor rotation applied to a single instrument.

**Entry**: SPY's 12-1 momentum is above its 63-day rolling median.

**Exit**: Momentum falls below median.

---

### 4. Regime Only

Long only when price > SMA(200) (or Markov score > 0). A baseline filter strategy.

---

## Risk Controls

| Control | Default |
|---|---|
| Risk per trade | 1% of equity |
| Reward/risk minimum | 2:1 |
| Daily loss circuit breaker | −2.5% |
| Max drawdown kill-switch | −8% |

---

## Anti-Overfitting Checklist

- [x] All signals shifted by 1 bar — no look-ahead bias
- [x] Walk-forward analysis (5 folds, 70/30 train/test)
- [x] Monte Carlo: 1,000 shuffles; PASS only if real Sharpe ∈ [median, 95th pct]
- [x] Parameter sensitivity sweep (Donchian) — edge must be robust across grid
- [x] Realistic costs: commission + slippage deducted on every position change

---

## Paper Trading Outline (disabled by default)

> ⚠️ Do not connect live trading without independent review, appropriate risk
> capital, and legal/regulatory compliance checks.

A paper trading extension would require:
1. **Live data feed**: Alpaca market data API (`alpaca-trade-api`) or Polygon.
2. **Signal generation**: same `strategies/` modules, run on a scheduler (e.g., `apscheduler`).
3. **Order execution**: Alpaca broker API — `api.submit_order(...)`.
4. **Alert channel**: Telegram bot (`python-telegram-bot`) for pre-order notifications.
5. **Human checkpoint**: orders held in a queue; human approves via `/approve` command before submission.
6. **Audit log**: append each signal + order to a local SQLite database.

No live-order code is included in this repository.

---

## Disclaimers

- Backtested / simulated results do not guarantee future performance.
- Past edge may not persist in live markets due to regime change, increased competition, or structural shifts.
- This tool is for research and educational purposes only. It is not investment advice.
- Always use risk capital you can afford to lose.
