"""
Market regime detection.

Implements the full markov-hedge-fund-method algorithm:
  - Rolling-return state labeling (Bull / Sideways / Bear)
  - MLE 3×3 transition matrix
  - Chapman-Kolmogorov n-step forecast (matrix power)
  - Stationary distribution (left eigenvector)
  - Signal = P(Bull) − P(Bear)
  - Optional HMM via hmmlearn (graceful fallback)
  - Walk-forward backtest of the regime signal itself

Reference: jackson-video-resources/markov-hedge-fund-method
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# A. Simple SMA trend filter
# ---------------------------------------------------------------------------

def sma_trend_filter(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Return boolean Series: True when Close > SMA(period). Shift(1) applied — no look-ahead."""
    sma = df["Close"].rolling(period).mean()
    return (df["Close"] > sma).shift(1).fillna(False)


# ---------------------------------------------------------------------------
# B. 3-state Markov regime model
# ---------------------------------------------------------------------------

_STATE_BULL = 0
_STATE_SIDE = 1
_STATE_BEAR = 2
_STATE_NAMES = {_STATE_BULL: "Bull", _STATE_SIDE: "Sideways", _STATE_BEAR: "Bear"}


def _label_states(daily_df: pd.DataFrame, fwd_period: int = 20, threshold: float = 0.05) -> pd.Series:
    """Label each bar as Bull/Sideways/Bear using forward returns (no look-ahead in backtest — used only for model fitting on train data)."""
    fwd_ret = daily_df["Close"].pct_change(fwd_period).shift(-fwd_period)
    labels = pd.Series(_STATE_SIDE, index=daily_df.index)
    labels[fwd_ret > threshold] = _STATE_BULL
    labels[fwd_ret < -threshold] = _STATE_BEAR
    return labels


def _build_transition_matrix(labels: pd.Series) -> np.ndarray:
    """Estimate 3×3 row-stochastic transition matrix from label sequence."""
    T = np.zeros((3, 3))
    for i in range(len(labels) - 1):
        s, s_next = int(labels.iloc[i]), int(labels.iloc[i + 1])
        T[s, s_next] += 1
    # Normalize rows; handle zero rows gracefully
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def _current_state_vector(label: int) -> np.ndarray:
    v = np.zeros(3)
    v[label] = 1.0
    return v


def markov_regime_score(
    daily_df: pd.DataFrame,
    fwd_period: int = 20,
    threshold: float = 0.05,
    n_steps: int = 1,
) -> pd.Series:
    """
    Return a continuous regime score Series (range -1 to +1).
    score = P(Bull) - P(Bear) projected n_steps forward.
    Built with a rolling expanding window to avoid look-ahead.
    """
    labels = _label_states(daily_df, fwd_period, threshold)
    scores = pd.Series(np.nan, index=daily_df.index)

    min_train = max(fwd_period * 3, 60)
    for i in range(min_train, len(daily_df)):
        train_labels = labels.iloc[:i]
        T = _build_transition_matrix(train_labels)
        Tn = np.linalg.matrix_power(T, n_steps)
        current = _current_state_vector(int(train_labels.iloc[-1]))
        probs = current @ Tn
        scores.iloc[i] = float(probs[_STATE_BULL] - probs[_STATE_BEAR])

    return scores.shift(1)  # shift so today's decision uses yesterday's score


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Left eigenvector for eigenvalue ≈ 1 — the long-run regime mix.
    From markov-hedge-fund-method/markov_regime.py.
    """
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.abs(np.real(eigvecs[:, idx]))
    total = vec.sum()
    return vec / total if total > 0 else vec


def regime_summary(daily_df: pd.DataFrame, window: int = 20, threshold: float = 0.05) -> dict:
    """
    Full Markov regime summary matching the markov-hedge-fund-method JSON output contract:
      current_regime, next_state_probabilities, signal,
      transition_matrix, persistence_diagonal, stationary_distribution, backtest
    """
    labels = _label_states(daily_df, window, threshold)
    valid = labels.dropna()
    if len(valid) < 60:
        return {"error": "Insufficient data for regime model (need ≥ 60 bars)."}

    T = _build_transition_matrix(valid)
    current_state = int(valid.iloc[-1])
    T1 = np.linalg.matrix_power(T, 1)
    probs = T1[current_state]

    stat = stationary_distribution(T)
    signal = float(probs[_STATE_BULL] - probs[_STATE_BEAR])

    # Walk-forward backtest of the regime signal itself
    bt_returns = []
    min_train = max(window * 3, 60)
    daily_ret = daily_df["Close"].pct_change().fillna(0)
    for i in range(min_train, len(daily_df) - 1):
        train = valid.iloc[:i]
        T_t = _build_transition_matrix(train)
        cs = int(train.iloc[-1])
        p = T_t[cs]
        sig = float(p[_STATE_BULL] - p[_STATE_BEAR])
        pos = 1 if sig > 0 else (-1 if sig < 0 else 0)
        bt_returns.append(pos * daily_ret.iloc[i + 1])

    bt_arr = np.array(bt_returns)
    bt_sharpe = float(bt_arr.mean() / bt_arr.std() * np.sqrt(252)) if bt_arr.std() > 0 else 0.0
    cum = np.cumprod(1 + bt_arr)
    peak = np.maximum.accumulate(cum)
    bt_mdd = float(((cum - peak) / peak).min()) if len(cum) > 0 else 0.0

    return {
        "current_regime": _STATE_NAMES[current_state],
        "next_state_probabilities": {
            "bear": round(float(probs[_STATE_BEAR]), 4),
            "sideways": round(float(probs[_STATE_SIDE]), 4),
            "bull": round(float(probs[_STATE_BULL]), 4),
        },
        "signal": round(signal, 4),
        "transition_matrix": T.round(4).tolist(),
        "persistence_diagonal": {
            "bear": round(float(T[_STATE_BEAR, _STATE_BEAR]), 4),
            "sideways": round(float(T[_STATE_SIDE, _STATE_SIDE]), 4),
            "bull": round(float(T[_STATE_BULL, _STATE_BULL]), 4),
        },
        "stationary_distribution": {
            "bear": round(float(stat[_STATE_BEAR]), 4),
            "sideways": round(float(stat[_STATE_SIDE]), 4),
            "bull": round(float(stat[_STATE_BULL]), 4),
        },
        "backtest": {
            "sharpe": round(bt_sharpe, 3),
            "max_drawdown": round(bt_mdd, 4),
            "n_trades": int((np.diff(np.sign(bt_arr)) != 0).sum()),
        },
    }


def hmm_regime_score(daily_df: pd.DataFrame, n_components: int = 3) -> pd.Series:
    """
    Hidden Markov Model regime score using hmmlearn.
    Falls back to GaussianMixture if hmmlearn unavailable.
    Returns continuous score aligned to [-1, +1].
    """
    returns = daily_df["Close"].pct_change().dropna().values.reshape(-1, 1)
    idx = daily_df.index[1:]

    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        # Map states to Bull/Bear/Sideways by mean return
        means = [returns[states == s].mean() for s in range(n_components)]
        bull_state = int(np.argmax(means))
        bear_state = int(np.argmin(means))
        score = pd.Series(0.0, index=idx)
        score[states == bull_state] = 1.0
        score[states == bear_state] = -1.0
    except ImportError:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=n_components, random_state=42)
        gm.fit(returns)
        labels = gm.predict(returns)
        means = [returns[labels == s].mean() for s in range(n_components)]
        bull_state = int(np.argmax(means))
        bear_state = int(np.argmin(means))
        score = pd.Series(0.0, index=idx)
        score[labels == bull_state] = 1.0
        score[labels == bear_state] = -1.0

    full = pd.Series(np.nan, index=daily_df.index)
    full.loc[idx] = score.values
    return full.shift(1).fillna(0)


def get_regime_signal(
    daily_df: pd.DataFrame,
    model: str = "sma",
    sma_period: int = 200,
    fwd_period: int = 20,
    threshold: float = 0.05,
    n_steps: int = 1,
) -> pd.Series:
    """
    Unified regime signal. Returns float Series:
      sma    → 1.0 (up) or 0.0 (down)
      markov → continuous in [-1, +1]
      hmm    → continuous in [-1, +1]
    """
    if model == "sma":
        return sma_trend_filter(daily_df, sma_period).astype(float)
    elif model == "markov":
        return markov_regime_score(daily_df, fwd_period, threshold, n_steps)
    elif model == "hmm":
        return hmm_regime_score(daily_df)
    else:
        raise ValueError(f"Unknown regime model: {model!r}. Choose 'sma', 'markov', or 'hmm'.")


def label_regime_states(daily_df: pd.DataFrame, regime_signal: pd.Series) -> pd.Series:
    """Convert numeric regime signal to string state labels for plotting."""
    states = pd.Series("Sideways", index=regime_signal.index)
    states[regime_signal > 0.1] = "Bull"
    states[regime_signal < -0.1] = "Bear"
    return states


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_regime(
    daily_df: pd.DataFrame,
    regime_signal: pd.Series,
    trades_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Price chart with shaded Bull/Bear/Sideways regime bands and optional trade markers."""
    states = label_regime_states(daily_df, regime_signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_df.index, y=daily_df["Close"],
        name="Price", line=dict(color="#4C72B0", width=1.5), mode="lines",
    ))

    color_map = {"Bull": "rgba(44,160,44,0.12)", "Sideways": "rgba(200,200,200,0.10)", "Bear": "rgba(214,39,40,0.12)"}

    # Draw shaded bands for each regime run
    prev_state = None
    band_start = None
    for ts, state in states.items():
        if state != prev_state:
            if prev_state is not None and band_start is not None:
                fig.add_vrect(
                    x0=band_start, x1=ts,
                    fillcolor=color_map.get(prev_state, "rgba(0,0,0,0)"),
                    layer="below", line_width=0,
                )
            band_start = ts
            prev_state = state
    if band_start is not None and prev_state is not None:
        fig.add_vrect(
            x0=band_start, x1=daily_df.index[-1],
            fillcolor=color_map.get(prev_state, "rgba(0,0,0,0)"),
            layer="below", line_width=0,
        )

    if trades_df is not None and not trades_df.empty:
        entries = trades_df[trades_df["entry_dt"].notna()]
        exits = trades_df[trades_df["exit_dt"].notna()]
        if not entries.empty:
            fig.add_trace(go.Scatter(
                x=entries["entry_dt"], y=entries["entry_px"],
                mode="markers", marker=dict(symbol="triangle-up", color="green", size=9),
                name="Entry",
            ))
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits["exit_dt"], y=exits["exit_px"],
                mode="markers", marker=dict(symbol="triangle-down", color="red", size=9),
                name="Exit",
            ))

    fig.update_layout(
        title="Price with Regime Bands",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h"),
        height=400, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
