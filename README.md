# Quant Trading App — Local Streamlit Backtester

A fully self-contained, locally runnable Streamlit application for systematic
trading strategy design, backtesting, and visualization across **S&P 500**,
**NASDAQ 100**, and **Crypto** (BTC, ETH, SOL, and more).

No paid API keys required. Data from Yahoo Finance (yfinance) and the Binance
public API, cached to disk.

---

## Local Deployment

The app runs entirely on your machine. No accounts, API keys, or paid data feeds are
required — market data comes from the free Yahoo Finance and Binance public endpoints.

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10 – 3.14** | Check with `python3 --version`. macOS/Linux ship Python 3; on Windows install from [python.org](https://www.python.org/downloads/). |
| **pip** | Bundled with modern Python. |
| **Internet connection** | Needed on first run to download price data. After that the on-disk cache lets you work offline (see [Offline Use](#offline-use--data-cache)). |
| **~200 MB free disk** | For the virtual environment and cached price data. |

### Step 1 — Get the code

```bash
git clone https://github.com/xozai/quant_app.git
cd quant_app
```

(If you already have the folder, just `cd quant_app`.)

### Step 2 — Create an isolated virtual environment

A virtual environment keeps the app's dependencies separate from your system Python.

```bash
python3 -m venv .venv
```

### Step 3 — Activate the environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat
```

Your shell prompt should now be prefixed with `(.venv)`. Re-run this command in every
new terminal session before launching the app.

### Step 4 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, pandas, numpy, yfinance, plotly, scipy, scikit-learn, hmmlearn,
pytest, and pyarrow. Installation takes 1–3 minutes.

### Step 5 — Launch the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (default **http://localhost:8501**) and opens it in your
browser automatically. If it doesn't open, paste the URL manually.

To run on a different port or expose it on your LAN:

```bash
streamlit run app.py --server.port 8600 --server.address 0.0.0.0
```

### Step 6 — First run

The app loads with sensible defaults (SPY, Donchian Breakout, daily bars, last ~5 years).
Click **🚀 Run Backtest** in the sidebar — results appear in a few seconds. The first
fetch for any ticker hits the network; subsequent runs read from cache and are instant.

### Stopping & restarting

- **Stop the server:** press `Ctrl+C` in the terminal running Streamlit.
- **Restart later:** re-activate the venv (Step 3) and re-run `streamlit run app.py`.
- **Deactivate the venv:** run `deactivate`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: streamlit` | The venv isn't activated, or deps aren't installed. Repeat Steps 3–4. |
| `ModuleNotFoundError: No module named 'pandas'` | Same as above — you're using system Python, not the venv. |
| Browser shows "connection refused" | The server isn't running, or the port is taken. Check the terminal, or try `--server.port 8600`. |
| Empty chart / `DataError` | The ticker symbol is wrong, the date range has no trading days, or Yahoo Finance is rate-limiting. Try again or pick a different ticker. |
| `hmmlearn` install fails | Optional — the HMM regime model falls back to a scikit-learn GaussianMixture automatically. The other models work without it. |

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

## How to Use the App

The interface is split into a **sidebar** (left — all inputs) and **ten tabs** (right —
all outputs). The workflow is always the same: set parameters in the sidebar → click
**🚀 Run Backtest** → read the results in the tabs.

### A. Configure your backtest (sidebar)

Work top to bottom:

1. **Market & Instrument**
   - **Market Preset** — pick S&P 500 (SPY), NASDAQ 100 (QQQ), Bitcoin, Ethereum, Solana,
     or **Custom ticker…**. Choosing Custom reveals a text box (e.g. `NVDA`, `AAPL`,
     `BTC-USD`) and an Asset Class toggle (equity vs crypto).
   - For crypto presets, a **Crypto Asset** dropdown lets you switch among the 10 supported coins.

2. **Strategy Family** — choose one of the five strategies (see
   [Strategy Families](#strategy-families) for the rules behind each):
   Donchian Breakout · VWAP+EMA+RSI Scalp · Trend-Join Momentum · Factor (Momentum) · Regime Only.

3. **Date range** — Start and End dates. Defaults to roughly the last five years.

4. **Timeframe** — Daily for all markets; intraday 15m/5m for equities, 1h/4h for crypto.
   (Intraday history from the free feed is limited to recent months.)

5. **Capital & Risk** — starting capital, risk-per-trade %, and an optional
   **Allow Short Positions** toggle (used by the scalp strategy).

6. **Regime Model** — the filter that gates when trades are allowed:
   - **SMA Trend Filter** — simplest; trade only when price > SMA(N).
   - **3-State Markov** — Bull/Sideways/Bear transition model with an n-step forecast.
   - **Hidden Markov (HMM)** — Gaussian HMM (auto-falls back to GaussianMixture).
   - Sub-sliders set the SMA period, Markov window, and Bull/Bear threshold.

7. **Strategy Parameters** — ATR period, Donchian period, ATR multiplier `k`, weekly
   confirmation toggle, and the RSI/VWAP knobs for the scalp strategy.

8. **Costs** — commission and slippage in basis points, applied on every position change.

9. Click **🚀 Run Backtest**.

### B. Read the results (tabs)

| Tab | What to look at |
|---|---|
| **📊 Results** | Start here. KPI tiles (CAGR, Sharpe, Sortino, Max DD, Win Rate, Profit Factor), the equity curve vs buy-and-hold, the drawdown chart, the **rolling Sharpe** chart (is the edge stable or only present in one period?), **benchmark metrics vs SPY** (alpha, beta, information ratio), the trade log, and the **CSV / HTML report** export buttons. |
| **⚖️ Compare** | Click **Compare All Strategies** to run all five on the same ticker and date range. Read the side-by-side metrics table and overlaid equity curves to pick the best fit. |
| **🔭 Universe Scan** | Pick a universe (NASDAQ 100 / S&P 500 / Crypto) and a strategy, then **Run Universe Scan** to rank every ticker by Sharpe. Download the ranked table as CSV. |
| **🗺️ Regime** | The current regime, next-state probabilities, the 3×3 transition matrix, the stationary distribution, and a price chart with shaded Bull/Sideways/Bear bands. |
| **🔬 Validation** | Out-of-sample checks: 5-fold walk-forward, a 1,000-sim Monte Carlo acceptance test (PASS / FAIL_OVERFIT), and a Donchian parameter heatmap. |
| **🧪 Strategy Audit** | A 6-test stress framework with PASS/FAIL badges per test and an overall verdict (STRONG / MARGINAL / FAIL). |
| **💰 Capital Allocator** | Three modes: **Kelly Criterion** sizing, **Multi-Strategy Weights** (Sharpe-weighted), and **Markowitz** (pick 2–5 assets → efficient frontier, max-Sharpe & min-vol portfolios, plus a **correlation heatmap**). |
| **📓 Journal** | Log real or paper trades, then review win rate, profit factor, and P&L by day-of-week. Persists to `~/.quant_app_cache/trade_journal.csv`. |
| **🤖 Agent Firm** | Reference spec for a 6-agent autonomous trading firm and an editable risk-thresholds table. |
| **ℹ️ About** | Full strategy spec, integrated repos, and disclaimers. |

### C. Recommended first walkthrough

1. **Results** — run the default SPY / Donchian backtest and read the KPI tiles.
2. **Validation** — confirm the Monte Carlo verdict is **PASS** (not FAIL_OVERFIT).
3. **Strategy Audit** — check the overall verdict and which of the 6 tests fail.
4. **Compare** — see whether another strategy beats Donchian on this ticker.
5. **Universe Scan** — find the strongest tickers in a universe for your chosen strategy.
6. **Capital Allocator** — size positions with Kelly and check cross-asset correlations.
7. **📄 Export HTML Report** (Results tab) — save a shareable summary.

### D. Share or bookmark a backtest

After you click Run Backtest, the app writes your settings (market preset, strategy,
dates, capital, regime model) into the page URL. **Copy the URL** to bookmark that exact
configuration or share it — opening it later restores those sidebar settings automatically.

### Offline Use & Data Cache

Every fetch is cached as a Parquet file under `~/.quant_app_cache/` and reused for 24 hours.
After the first online run for a ticker/timeframe, you can re-run that backtest with no
network connection. To force fresh data, delete the relevant file (or the whole folder):

```bash
rm -rf ~/.quant_app_cache        # macOS / Linux
# Windows: rmdir /s %USERPROFILE%\.quant_app_cache
```

> ⚠️ This tool is for research and education only — it is **not** investment advice. See
> the [Disclaimers](#disclaimers) at the end of this README.

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
│   ├── backtest.py               Vectorized engine · rolling Sharpe · benchmark metrics
│   ├── risk.py                   Fixed-fractional sizing · circuit breakers
│   ├── validation.py             Walk-forward · Monte Carlo · 6-test audit
│   ├── capital_allocator.py      Kelly criterion · MVO · multi-strategy weights
│   ├── scanner.py                Universe scan — rank tickers by Sharpe
│   ├── report.py                 Self-contained HTML report generator
│   └── journal.py                Trade logging · performance review
├── tests/
│   ├── test_smoke.py             22 pytest smoke tests
│   └── test_comprehensive.py     73 module-level tests
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
