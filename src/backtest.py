"""
Historical backtest of the delta-hedging strategy from hedging.py, run
against REAL historical price paths instead of simulated GBM paths.

Motivation: simulate_delta_hedge (hedging.py) tests the hedging
strategy's mechanics against paths generated under the SAME model the
hedger uses to compute delta (GBM with constant vol) - that proves the
implementation is correct, but says nothing about how the strategy
performs against real markets, which have volatility clustering, fat
tails, and regime changes that GBM doesn't capture. This module closes
that gap: identical hedging mechanics, but the underlying path is actual
historical daily closes, not a random draw from the hedger's own model.

Design - rolling-window backtest: for a chosen window length (e.g. 63
trading days ~ one quarter), every possible start date in the historical
sample defines one "trial": sell an ATM option struck at that day's
price, using volatility estimated ONLY from data strictly before that
day (a trailing realized-vol lookback - critical to avoid lookahead
bias, since a real hedger at inception cannot see future returns), then
delta-hedge using that same fixed vol assumption across the window on
the REAL subsequent price path, rebalancing at a fixed frequency. This
produces one hedging P&L per rolling window; across all windows in the
sample, that's a real-market analogue to the Monte Carlo P&L
distribution in hedging.py - same strategy, real paths instead of
simulated ones.

Important caveat (see README Limitations): rolling windows overlap
(window i and window i+5 share most of their days), so these P&L
outcomes are NOT independent draws the way Monte Carlo paths are - the
resulting distribution understates true uncertainty and should be read
as a directional/qualitative comparison against the model-simulated
distribution, not a literal confidence interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing.black_scholes import BSInputs, OptionType, bs_price
from .greeks import compute_greeks

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    pnl: np.ndarray             # one hedging P&L per rolling window
    vols_used: np.ndarray       # trailing realized vol assumed at each window's inception
    window_days: int
    rebalance_every: int

    @property
    def n_windows(self) -> int:
        return len(self.pnl)

    @property
    def mean(self) -> float:
        return float(np.mean(self.pnl)) if len(self.pnl) else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.pnl, ddof=1)) if len(self.pnl) > 1 else float("nan")


def estimate_realized_vol(prices: np.ndarray, lookback: int = 60) -> float:
    """Annualized realized volatility from trailing daily log returns.

    Args:
        prices: Price array, most recent price last. Only the trailing
            `lookback` returns (lookback + 1 prices) are used.
        lookback: Number of trailing daily returns to use.

    Returns:
        Annualized volatility (e.g. 0.20 for 20%).
    """
    window = prices[-(lookback + 1):]
    log_returns = np.diff(np.log(window))
    return float(np.std(log_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def backtest_delta_hedge(
    prices: pd.Series | np.ndarray,
    window_days: int = 60,
    rebalance_every: int = 5,
    vol_lookback: int = 60,
    r: float = 0.045,
    option_type: OptionType = OptionType.CALL,
    moneyness: float = 1.0,
) -> BacktestResult:
    """Roll a delta-hedged option position across real historical price windows.

    Args:
        prices: Daily close prices, chronologically ordered.
        window_days: Length of each hedge window in trading days (e.g.
            60 ~ one quarter). Should be evenly divisible by
            rebalance_every - if not, the final partial period is held
            to expiry without an intermediate rebalance.
        rebalance_every: Rebalance the hedge every this many trading days.
        vol_lookback: Trading days of trailing history used to estimate
            the volatility assumed at each window's inception (strictly
            before the window starts - no lookahead bias).
        r: Risk-free rate, held constant across the whole sample (real
            rates obviously vary across any multi-year window - see
            README Limitations).
        option_type: CALL or PUT.
        moneyness: Strike as a multiple of the window's starting price
            (1.0 = struck at-the-money at inception).

    Returns:
        BacktestResult with one hedging P&L per rolling window.
    """
    values = np.asarray(prices, dtype=float)
    n = len(values)
    T = window_days / TRADING_DAYS_PER_YEAR

    earliest_start = vol_lookback
    latest_start = n - window_days - 1
    if latest_start < earliest_start:
        raise ValueError(
            f"Not enough price history: need at least "
            f"{vol_lookback + window_days + 1} data points, got {n}."
        )

    pnl_list = []
    vol_list = []

    for start in range(earliest_start, latest_start + 1):
        S0 = values[start]
        K = S0 * moneyness
        sigma = estimate_realized_vol(values[: start + 1], lookback=vol_lookback)
        if sigma <= 1e-6:
            continue  # degenerate window (e.g. a data gap); skip rather than divide by ~0

        premium = bs_price(BSInputs(S=S0, K=K, T=T, r=r, sigma=sigma), option_type)
        cash = premium
        delta_prev = 0.0

        for day_offset in range(0, window_days, rebalance_every):
            t = day_offset / TRADING_DAYS_PER_YEAR
            time_to_expiry = max(T - t, 1e-8)
            S_t = values[start + day_offset]

            inputs = BSInputs(S=S_t, K=K, T=time_to_expiry, r=r, sigma=sigma)
            delta_t = compute_greeks(inputs, option_type).delta

            trade = delta_t - delta_prev
            cash -= trade * S_t
            cash *= math.exp(r * (rebalance_every / TRADING_DAYS_PER_YEAR))
            delta_prev = delta_t

        S_T = values[start + window_days]
        payoff = max(S_T - K, 0.0) if option_type == OptionType.CALL else max(K - S_T, 0.0)
        portfolio_value = cash + delta_prev * S_T - payoff

        pnl_list.append(portfolio_value)
        vol_list.append(sigma)

    return BacktestResult(
        pnl=np.array(pnl_list),
        vols_used=np.array(vol_list),
        window_days=window_days,
        rebalance_every=rebalance_every,
    )
