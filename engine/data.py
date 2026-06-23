"""Data fetching and on-disk caching via yfinance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".quant_app_cache"
CACHE_TTL_HOURS = 24

# ---------------------------------------------------------------------------
# Curated universes
# ---------------------------------------------------------------------------

NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "QCOM", "TMUS", "INTC", "INTU", "AMGN",
    "TXN", "AMAT", "MU", "BKNG", "ISRG", "LRCX", "KLAC", "PANW", "SNPS",
    "CDNS", "ASML", "REGN", "MDLZ", "ADP", "GILD", "CSX", "VRTX", "MELI",
    "PYPL", "CHTR", "CMCSA", "MNST", "ORLY", "IDXX", "PCAR", "KDP", "EXC",
    "DXCM", "CTAS", "BIIB", "FAST", "TEAM",
]

CRYPTO_TICKERS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "BNB (BNB-USD)": "BNB-USD",
    "XRP (XRP-USD)": "XRP-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Avalanche (AVAX-USD)": "AVAX-USD",
    "Dogecoin (DOGE-USD)": "DOGE-USD",
    "Chainlink (LINK-USD)": "LINK-USD",
    "Polkadot (DOT-USD)": "DOT-USD",
}

SP500_TICKERS = [
    "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY",
    "JPM", "AVGO", "TSLA", "UNH", "XOM", "V", "MA", "JNJ", "PG", "HD",
    "MRK", "COST", "CVX", "ABBV", "BAC", "WMT", "CRM", "AMD", "TMO", "ACN",
    "LIN", "MCD", "KO", "PEP", "ADBE", "DHR", "ABT", "CSCO", "WFC", "TXN",
    "INTU", "NEE", "AMGN", "MS", "RTX", "GS", "UPS", "AMAT", "SPGI", "ISRG",
]

MARKET_PRESETS = {
    "S&P 500 — SPY": ("SPY", "equity"),
    "NASDAQ 100 — QQQ": ("QQQ", "equity"),
    "Bitcoin — BTC-USD": ("BTC-USD", "crypto"),
    "Ethereum — ETH-USD": ("ETH-USD", "crypto"),
    "Solana — SOL-USD": ("SOL-USD", "crypto"),
    "Custom ticker…": (None, None),
}

# Crypto trades 24/7 — no session gaps; flag for downstream handling
CRYPTO_TICKERS_SET = set(CRYPTO_TICKERS.values())


def is_crypto(ticker: str) -> bool:
    return ticker.upper().endswith("-USD") or ticker.upper().endswith("USDT")


# ---------------------------------------------------------------------------
# Cache + fetch
# ---------------------------------------------------------------------------


class DataError(Exception):
    pass


def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    key = f"{ticker}_{interval}_{start}_{end}"
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{ticker}_{interval}_{safe}.parquet"


def fetch(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return OHLCV DataFrame. Parquet cache; refreshes after 24 h."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, interval, start, end)

    if path.exists():
        age_hours = (
            pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
        ).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            df = pd.read_parquet(path)
            if not df.empty:
                return df

    try:
        raw = yf.download(
            ticker, start=start, end=end, interval=interval,
            progress=False, auto_adjust=True,
        )
    except Exception as exc:
        raise DataError(f"Network error fetching {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(
            f"No data returned for {ticker} ({interval}) from {start} to {end}. "
            "Check the ticker symbol and date range."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    df.to_parquet(path)
    return df


def fetch_daily_for_regime(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Always returns daily data for regime/SMA computation."""
    return fetch(ticker, start, end, interval="1d")


def fetch_binance_ohlcv(symbol: str, interval: str = "4h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance public API (no auth needed).
    symbol examples: 'BTCUSDT', 'ETHUSDT'
    interval: '1m','5m','15m','1h','4h','1d'
    """
    import urllib.request, json

    binance_symbol = symbol.replace("-", "").upper()
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={binance_symbol}&interval={interval}&limit={limit}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise DataError(f"Binance fetch failed for {symbol}: {exc}") from exc

    if not data:
        raise DataError(f"No Binance data for {symbol}.")

    df = pd.DataFrame(data, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df.index.name = "Datetime"
    return df[["Open", "High", "Low", "Close", "Volume"]]
