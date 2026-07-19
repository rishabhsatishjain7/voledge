"""
Market data fetching and cleaning for the options pricing engine.

Source: Yahoo Finance via `yfinance`. Free and convenient, but the options
chain data has known quality issues that this module filters for explicitly
rather than silently trusting:
    - Stale quotes on illiquid strikes (last trade could be days old)
    - Zero volume / zero open interest strikes with meaningless mid-prices
    - Crossed or wide bid-ask spreads
    - Bid/ask/openInterest fields sometimes come back entirely zeroed out
      for a given pull even when volume is clearly real (a known Yahoo/
      yfinance data-quality gap) - handled via a lastPrice fallback below.

See README "Limitations" for the full caveat on data quality and what
this filtering does and does not catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    expiry: str
    bid: float
    ask: float
    last_price: float
    volume: float
    open_interest: float
    implied_vol_yahoo: float  # Yahoo's own IV, for cross-checking our solver

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0


def get_spot_price(ticker: str) -> float:
    """Fetch the current (delayed) spot price for a ticker."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    return float(hist["Close"].iloc[-1])


def get_price_history(ticker: str, period: str = "3y") -> pd.Series:
    """Fetch daily closing prices for a ticker, for use in backtest.py.

    Args:
        ticker: Underlying ticker symbol, e.g. "AAPL".
        period: yfinance period string (e.g. "1y", "3y", "5y", "max").

    Returns:
        Chronologically ordered pandas Series of daily close prices,
        indexed by date. NaN rows (e.g. from trading halts) are dropped.
    """
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No price history returned for ticker '{ticker}'.")
    return hist["Close"].dropna()


def get_available_expiries(ticker: str) -> list[str]:
    """List available options expiry dates for a ticker."""
    return list(yf.Ticker(ticker).options)


def _clean_chain(
    df: pd.DataFrame,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: float,
    max_quote_age_days: int = 3,
) -> pd.DataFrame:
    """Filter a raw options-chain DataFrame down to usable, liquid rows.

    Decided PER ROW, not per chain: a mixed chain (some strikes with real
    bid/ask, most without - the common case on this data source) needs
    each strike judged on its own quote quality, not funneled into a
    single chain-wide strategy.

    For each row:
      - If it has a real bid/ask (bid > 0, ask > bid) with an acceptable
        relative spread: use the bid/ask midpoint as `mid`.
      - Otherwise: fall back to lastPrice as `mid`.

    Either way, the row is also required to have traded within
    max_quote_age_days. This matters even for rows with a real-looking
    bid/ask: a bid/ask pair sitting on a strike that hasn't traded in
    months can itself be stale/wide/erroneous, producing a mid price
    wildly out of line with neighboring strikes (observed in practice -
    see README Limitations).
    """
    df = df.copy()
    has_quotes = (df["bid"] > 0) & (df["ask"] > df["bid"])
    df["mid"] = df["lastPrice"].astype(float)
    df.loc[has_quotes, "mid"] = ((df["bid"] + df["ask"]) / 2.0)[has_quotes]

    last_trade = pd.to_datetime(df["lastTradeDate"], utc=True)
    age_days = (pd.Timestamp.now(tz="UTC") - last_trade).dt.total_seconds() / 86400.0
    fresh = age_days <= max_quote_age_days

    spread_ok = has_quotes & (((df["ask"] - df["bid"]) / df["mid"]) <= max_spread_pct)

    quote_row_mask = (
        has_quotes
        & spread_ok
        & (df["volume"].fillna(0) >= min_volume)
        & (df["openInterest"].fillna(0) >= min_open_interest)
        & fresh
    )
    fallback_row_mask = (
        (~has_quotes)
        & (df["volume"].fillna(0) >= max(min_volume, 1))
        & fresh
    )

    mask = quote_row_mask | fallback_row_mask
    return df[mask].reset_index(drop=True)


def get_option_chain(
    ticker: str,
    expiry: str,
    min_volume: int = 1,
    min_open_interest: int = 1,
    max_spread_pct: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and clean the options chain for a given ticker and expiry.

    Args:
        ticker: Underlying ticker symbol, e.g. "AAPL".
        expiry: Expiry date string as returned by get_available_expiries.
        min_volume: Minimum daily volume to keep a strike.
        min_open_interest: Minimum open interest to keep a strike.
        max_spread_pct: Maximum allowed relative bid-ask spread.

    Returns:
        (calls, puts) as cleaned pandas DataFrames.
    """
    chain = yf.Ticker(ticker).option_chain(expiry)
    calls = _clean_chain(chain.calls, min_volume, min_open_interest, max_spread_pct)
    puts = _clean_chain(chain.puts, min_volume, min_open_interest, max_spread_pct)
    return calls, puts


def option_row_to_quote(row: pd.Series, expiry: str) -> OptionQuote:
    """Convert a cleaned DataFrame row into an OptionQuote."""
    return OptionQuote(
        strike=float(row["strike"]),
        expiry=expiry,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        last_price=float(row["lastPrice"]),
        volume=float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
        open_interest=float(row["openInterest"]) if pd.notna(row["openInterest"]) else 0.0,
        implied_vol_yahoo=float(row["impliedVolatility"]),
    )


def years_to_expiry(expiry: str) -> float:
    """Convert an expiry date string ('YYYY-MM-DD') to years from today."""
    expiry_date = pd.Timestamp(expiry)
    today = pd.Timestamp.now().normalize()
    days = (expiry_date - today).days
    return max(days, 0) / 365.0
