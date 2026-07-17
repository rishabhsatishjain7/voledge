"""
Market data fetching and cleaning for the options pricing engine.

Source: Yahoo Finance via `yfinance`. Free and convenient, but the options
chain data has known quality issues that this module filters for explicitly
rather than silently trusting:
    - Stale quotes on illiquid strikes (last trade could be days old)
    - Zero volume / zero open interest strikes with meaningless mid-prices
    - Crossed or wide bid-ask spreads

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


def get_available_expiries(ticker: str) -> list[str]:
    """List available options expiry dates for a ticker."""
    return list(yf.Ticker(ticker).options)


def get_option_chain(
    ticker: str,
    expiry: str,
    min_volume: int = 1,
    min_open_interest: int = 1,
    max_spread_pct: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and clean the options chain for a given ticker and expiry.

    Filters applied (see module docstring for rationale):
        - Drops strikes with volume < min_volume
        - Drops strikes with open interest < min_open_interest
        - Drops strikes where (ask - bid) / mid > max_spread_pct

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
    calls, puts = chain.calls.copy(), chain.puts.copy()

    for df in (calls, puts):
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (df["volume"].fillna(0) >= min_volume)
            & (df["openInterest"].fillna(0) >= min_open_interest)
            & (df["bid"] > 0)
            & (df["ask"] > df["bid"])
            & (((df["ask"] - df["bid"]) / df["mid"]) <= max_spread_pct)
        )
        return df[mask].reset_index(drop=True)

    return _clean(calls), _clean(puts)


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
