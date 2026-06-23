# Quant Trading App — Local Streamlit Backtester

A fully self-contained, locally runnable Streamlit application for systematic
trading strategy design, backtesting, and visualization across **S&P 500**,
**NASDAQ 100**, and **Crypto** (BTC, ETH, SOL, and more).

No paid API keys required. Data from Yahoo Finance (yfinance) and the Binance
public API, cached to disk.

---

## Quick Start

```bash
cd quant_app
python3 -m venv .venv
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

95 tests (22 smoke + 73 comprehensive) cover data fetching (equities + crypto), all strategy
signal generators, the backtest engine, rolling Sharpe, benchmark metrics, universe scanner,
correlation matrix properties, query param helpers, regime labeling, Monte Carlo structure,
Kelly criterion, capital allocator, and the 6-test strategy audit framework.

---

## Markets Supported

| Market | Ticker Examples | Data Source |
|---|---|---|
| S&P 500 | SPY, and top-50 constituents | Yahoo Finance |
| NASDAQ 100 | QQQ, and top-50 constituents | Yahoo Finance |
| Crypto | BTC-USD, ETH-USD, SOL-USD, + 7 more | Yahoo Finance + Binance public API |

Crypto data is available 24/7 with no session gaps. Intraday intervals (1h, 4h)
are supported for crypto; 15m and 5m for equities.

---

## Project Structure

```
quant_app/
├── app.py                        Streamlit UI (10 tabs)
├── strategies/
│   ├── regime.py                 SMA filter · 3-state Markov · HMM
│   ├── donchian.py               Donchian channel breakout (default)
│   ├── scalp.py                  VWAP + EMA(8) + RSI(3) scalping
│   ├── trend_join.py             Multi-confirmation trend-join entry
│   ├── factor.py                 Price-based momentum factor signal
│   └── opening_range.py          Opening-range scalp (intraday)
├── engine/
│   ├── data.py                   Fetch + parquet cache · market universes
│   ├── backtest.py               Vectorized engine with costs & slippage
│   ├── risk.py                   Fixed-fractional sizing · circuit breakers
│   ├── validation.py             Walk-forward · Monte Carlo · 6-test audit
│   ├── capital_allocator.py      Kelly criterion · MVO · multi-strategy weights
│   └── journal.py                Trade logging · performance review
├── tests/
│   └── test_smoke.py             17 pytest smoke tests
├── .github/workflows/
│   ├── pr-title.yml              Conventional Commits PR title linting
│   └── release-please.yml        Automated SemVer releases
├── release-please-config.json
├── .release-please-manifest.json
├── CONTRIBUTING.md
├── CHANGELOG.md
└── requirements.txt
```

---

## UI Tabs

| Tab | Contents |
|---|---|
| **📊 Results** | KPI tiles · equity curve · drawdown chart · rolling Sharpe chart · benchmark metrics vs SPY (alpha, beta, IR) · trade log + CSV export · HTML report export |
| **⚖️ Compare** | Run all 5 strategies on the same ticker — side-by-side metrics table and overlaid equity curves |
| **🔭 Universe Scan** | Batch-run any strategy across NASDAQ 100, S&P 500, or Crypto universe — ranked by Sharpe with CSV download |
| **🗺️ Regime** | Current regime · next-state probabilities · 3×3 transition matrix · persistence diagonal · stationary distribution · price chart with shaded bands |
| **🔬 Validation** | Walk-forward (5 folds) · Monte Carlo acceptance test (1,000 sims) · Donchian parameter sensitivity heatmap |
| **🧪 Strategy Audit** | 6-test stress framework: in-sample · walk-forward · Monte Carlo · parameter sensitivity · cost stress · drawdown analysis |
| **💰 Capital Allocator** | Kelly criterion · multi-strategy Sharpe-weighted allocation · Markowitz mean-variance optimizer with efficient frontier · pairwise correlation heatmap · strategy vs universe correlation |
| **📓 Journal** | Log trades · performance review (win rate, Sharpe, profit factor, P&L by day-of-week) · CSV export |
| **🤖 Agent Firm** | 6-agent autonomous trading firm spec · risk-thresholds editor · Paperclip API patterns · safety-check log schema |
| **ℹ️ About** | Full strategy spec · integrated repos · disclaimers |

---

## Strategy Families

### 1. Donchian Channel Breakout (default)

**Thesis**: Price breaking above its recent high signals trend initiation. Weekly
confirmation filters noise; ATR trailing stop rides the trend.

- **Entry**: Close > prior Donchian upper band. Optional weekly confirmation.
- **Exit**: Close < prior close − k × ATR(14).
- **Regime gate**: Price > SMA(200) or Markov score > 0.
- **Parameters**: `period` (default 20), `atr_period` (14), `k` (2.0).
- **Known failure modes**: Choppy markets cause whipsaws; tight ATR stop during
  high-volatility events; weekly confirmation can miss fast breakouts.

---

### 2. VWAP + EMA(8) + RSI(3) Scalp

**Thesis**: Three indicators, each with a distinct role — VWAP for session bias,
EMA(8) for trend direction, RSI(3) for entry timing. All must align before a trade.

- **Entry long**: Price > VWAP AND Price > EMA(8) AND RSI(3) < 30 AND within 1.5% of VWAP.
- **Entry short** (optional): Price < VWAP AND Price < EMA(8) AND RSI(3) > 70.
- **Exit**: RSI(3) crosses 50, or 0.3% stop hit.
- **Works on**: Crypto (24/7), equities (intraday or daily).
- **Safety-check gate**: All conditions logged to `safety_check_log.json` on every bar.

---

### 3. Trend-Join Momentum

**Thesis**: Enter only after multiple trend confirmations align simultaneously,
reducing false entries and focusing on high-conviction continuations.

- **Entry**: Close > prior High AND prior Close > SMA(200) AND gap > 5% AND
  volume > 50k AND price > $3.
- **Exit**: ATR trailing stop; or regime flip to negative.

---

### 4. Factor (Momentum)

**Thesis**: SPY's 12-1 month momentum relative to its rolling 63-day median
predicts near-term direction.

- **Entry**: 12-1 month momentum above rolling median.
- **Exit**: Momentum falls below median.

---

### 5. Regime Only

Long only when price > SMA(200) or Markov/HMM score > 0. Useful as a baseline
to isolate the value of the regime filter alone.

---

## Regime Models

| Model | Method | Signal Range |
|---|---|---|
| SMA Trend Filter | Close > SMA(N) | 0 or 1 |
| 3-State Markov | MLE transition matrix + Chapman-Kolmogorov n-step forecast | −1 to +1 |
| Hidden Markov (HMM) | Gaussian HMM via hmmlearn (falls back to GaussianMixture) | −1 to +1 |

The Markov tab outputs the full analysis contract: current regime, next-state
probabilities, transition matrix, persistence diagonal, stationary distribution,
and a walk-forward backtest of the regime signal itself.

---

## Risk Controls

| Control | Default |
|---|---|
| Risk per trade | 1% of equity |
| Reward/risk minimum | 2:1 |
| Daily loss circuit breaker | −2.5% |
| Max drawdown kill-switch | −8% |

---

## Validation Framework

### Walk-Forward
5 rolling folds (70% train / 30% test). Mean OOS Sharpe flagged if below 0.5.

### Monte Carlo Acceptance Test
1,000 random shuffles of trade P&L. Strategy **PASS**es only if real Sharpe ∈
[median simulation, 95th percentile]. A result above the 95th percentile is
flagged as **FAIL_OVERFIT**.

### 6-Test Strategy Audit
Full stress test from the `strategy-audit` skill:

| # | Test | Pass Threshold |
|---|---|---|
| 1 | In-sample | Sharpe > 1.0, Max DD < 20%, ≥ 50 trades |
| 2 | Walk-forward OOS | Mean OOS Sharpe ≥ 1.0 |
| 3 | Monte Carlo | Verdict = PASS |
| 4 | Parameter sensitivity | Sharpe drop < 30% on ±20% cost change |
| 5 | Cost stress (2× / 5×) | Sharpe > 0 at all cost multiples |
| 6 | Drawdown analysis | Max drawdown < 20% |

Verdict: **STRONG** (5–6 pass) · **MARGINAL** (3–4) · **FAIL** (< 3).

### Anti-Overfitting Checklist
- [x] All signals use `.shift(1)` — zero look-ahead bias
- [x] Walk-forward: 5 folds, 70/30
- [x] Monte Carlo: 1,000 shuffles, PASS ∈ [median, 95th pct]
- [x] 6-test strategy audit
- [x] Donchian parameter sensitivity heatmap (5 periods × 3 k-values)
- [x] Realistic costs: commission + slippage on every position change

---

## Capital Allocator

Three modes accessible from the **💰 Capital Allocator** tab:

- **Kelly Criterion** — computes full and half-Kelly position size from win rate,
  avg win, and avg loss.
- **Multi-Strategy Weights** — Sharpe-weighted allocation across up to 6 strategies
  with per-strategy Kelly sizing.
- **Markowitz MVO** — Monte Carlo efficient frontier for 2–5 assets; outputs
  max-Sharpe and min-volatility portfolios.

---

## Trade Journal

Persistent CSV log at `~/.quant_app_cache/trade_journal.csv`. Log trades from
the **📓 Journal** tab; performance review computes win rate, Sharpe, profit
factor, max consecutive losses, max drawdown, and P&L by day of week.

---

## Agent Firm Architecture (Reference)

The **🤖 Agent Firm** tab documents a 6-agent autonomous trading organization
based on the [paperclip-zero-human-trading-firm](https://github.com/jackson-video-resources/paperclip-zero-human-trading-firm)
pattern:

| Agent | Role |
|---|---|
| CEO | Coordinator; routes tasks; human approval gate |
| Research | Nightly strategy discovery |
| Backtest | Historical validation (Sharpe ≥ 1.5 required) |
| Risk Manager | Pre-execution gatekeeper; enforces `risk-thresholds.json` |
| Execution | Places trades (paper by default) |
| Cost Optimizer | Weekly token-spend audit |

No live orders are placed from this app.

---

## Changelog Highlights

### v0.4.0 — MVP P2 Features (2026-06-23)
- **Correlation heatmap** in Capital Allocator — pairwise return correlations among selected assets, plus strategy vs universe comparison (closes #18)
- **Persistent URL query params** — ticker, strategy, date range, capital, regime model written to URL on Run Backtest; paste URL to restore the exact backtest (closes #17)
- 2 new tests (`test_correlation_matrix_properties`, `test_query_params_helpers`)

### v0.3.0 — MVP P1 Features (2026-06-23)
- **Compare tab**: run all 5 strategies side-by-side on the same ticker
- **Universe Scan tab**: batch-rank NASDAQ 100, S&P 500, or Crypto universe by Sharpe
- **Rolling Sharpe chart**: visualize edge stability over configurable windows in Results tab
- **Benchmark metrics**: alpha, beta, information ratio, tracking error vs SPY
- **HTML report export**: one-click self-contained backtest summary shareable without the app
- 3 new tests (`test_rolling_sharpe_length`, `test_benchmark_metrics_keys`, `test_scanner_returns_dataframe`)

### v0.2.0 — Bug Fixes (2026-06-23)
- Fixed `position_size()` returning 1 (not 0) when stop == entry ([#4](https://github.com/xozai/quant_app/issues/4))
- Fixed short trade PnL sign in `journal.log_trade()` ([#5](https://github.com/xozai/quant_app/issues/5))
- Renamed Donchian output columns to `upper_band`/`lower_band` ([#6](https://github.com/xozai/quant_app/issues/6))
- Fixed `label_regime_states()` signature for direct labeling ([#7](https://github.com/xozai/quant_app/issues/7))
- Lowered trend-join gap filter default to 0.0 for ETF compatibility ([#8](https://github.com/xozai/quant_app/issues/8))
- Replaced deprecated `pd.Timestamp.utcnow()` ([#9](https://github.com/xozai/quant_app/issues/9))
- Added 73 comprehensive tests across all modules

---

## Integrated Repos

| Repo | Contribution |
|---|---|
| [`claude-tradingview-mcp-trading`](https://github.com/jackson-video-resources/claude-tradingview-mcp-trading) | VWAP+EMA+RSI scalp strategy; safety-check gate; trade CSV logging |
| [`markov-hedge-fund-method`](https://github.com/jackson-video-resources/markov-hedge-fund-method) | Full Markov regime model with stationary distribution and HMM option |
| [`paperclip-zero-human-trading-firm`](https://github.com/jackson-video-resources/paperclip-zero-human-trading-firm) | 6-agent firm spec; risk-thresholds schema; Paperclip API patterns |
| [`skills`](https://github.com/jackson-video-resources/skills) | 6-test strategy audit; Kelly/MVO capital allocator; trade journal schema |

---

## Paper Trading Outline (disabled — reference only)

> ⚠️ Do not connect live trading without independent review, appropriate risk
> capital, and legal/regulatory compliance checks.

1. **Live data**: Binance public API or Alpaca market data.
2. **Signal generation**: Same `strategies/` modules on a cron schedule.
3. **Safety check**: `strategies/scalp.safety_check()` must return `allowed=True`.
4. **Human checkpoint**: Signal queued; human approves via Telegram `/approve`.
5. **Execution**: Alpaca `api.submit_order(...)` (paper account first).
6. **Audit log**: Every decision appended to `safety_check_log.json` and `trade_journal.csv`.

---

## Releases

This repo uses [Conventional Commits](https://www.conventionalcommits.org/) and
[release-please](https://github.com/googleapis/release-please) for automated
SemVer versioning. See [CONTRIBUTING.md](CONTRIBUTING.md) for PR title rules.

---

## Disclaimers

- Backtested / simulated results do not guarantee future performance.
- Crypto markets are highly volatile — drawdowns can be severe and rapid.
- Past edge may not persist due to regime change, increased competition, or
  structural market shifts.
- This tool is for research and educational purposes only. It is **not** investment advice.
- Always use risk capital you can afford to lose.
