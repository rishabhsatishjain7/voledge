"""
Delta-hedging simulation: sell one option, dynamically hedge the delta
exposure with the underlying, and measure the P&L distribution at expiry.

Core result this module is built to produce (see notebooks/analysis.ipynb):
hedging error variance shrinks as rebalancing frequency increases. This is
the classic, robust result — always true under the model's own assumptions
(GBM, no transaction costs, continuous-time BS delta) — as opposed to
results that depend on getting the vol-misspecification setup exactly
right. It's the reliable headline number for this project.

Mechanics per path:
    1. Sell the option, receive the BS premium.
    2. At each rebalancing step, compute BS delta and adjust the
       underlying position to match it, financing/investing the cash
       difference at the risk-free rate.
    3. At expiry, settle the option payoff and unwind the hedge.
    4. Hedging P&L = final portfolio value (should be ~0 in a perfect
       continuous-time hedge; discrete rebalancing leaves residual error).

Ignores transaction costs and bid-ask spread by default (see README
Limitations) — set `transaction_cost_bps` to explore their effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pricing.black_scholes import BSInputs, OptionType, bs_price


def _vectorized_bs_delta(
    S: np.ndarray, K: float, T: float, r: float, sigma: float, q: float, option_type: OptionType
) -> np.ndarray:
    """Vectorized Black-Scholes delta across an array of spot prices.

    Mirrors greeks.compute_greeks but operates on a full path array at
    once, since simulate_delta_hedge needs a delta per path per
    rebalancing step and a Python-level loop there would be the
    performance bottleneck for n_paths in the thousands.
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return (S > K).astype(float)
        return -(S < K).astype(float)

    from scipy.stats import norm  # local import: only the hedging sim needs vectorized CDF

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    disc_q = math.exp(-q * T)
    ncdf = norm.cdf(d1)

    if option_type == OptionType.CALL:
        return disc_q * ncdf
    return disc_q * (ncdf - 1.0)


@dataclass(frozen=True)
class HedgeSimResult:
    pnl: np.ndarray  # one hedging P&L per simulated path
    n_rebalances: int

    @property
    def mean(self) -> float:
        return float(np.mean(self.pnl))

    @property
    def std(self) -> float:
        return float(np.std(self.pnl, ddof=1))


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_rebalances: int,
    option_type: OptionType = OptionType.CALL,
    n_paths: int = 5_000,
    q: float = 0.0,
    transaction_cost_bps: float = 0.0,
    seed: int | None = None,
) -> HedgeSimResult:
    """Simulate discrete delta hedging of a short option position.

    Args:
        S0, K, T, r, sigma, q: Standard Black-Scholes inputs (real-world
            drift is assumed equal to r, i.e. paths are simulated under
            the risk-neutral measure — a modeling simplification).
        n_rebalances: Number of equally-spaced rebalancing points over
            [0, T]. Higher = finer hedging, lower residual error.
        option_type: CALL or PUT.
        n_paths: Number of simulated underlying price paths.
        transaction_cost_bps: Round-trip cost per rebalancing trade, in
            basis points of trade notional. 0 = frictionless (default).
        seed: RNG seed for reproducibility.

    Returns:
        HedgeSimResult containing the per-path hedging P&L array.
    """
    if n_rebalances < 1:
        raise ValueError("n_rebalances must be >= 1.")

    rng = np.random.default_rng(seed)
    dt = T / n_rebalances

    inputs0 = BSInputs(S=S0, K=K, T=T, r=r, sigma=sigma, q=q)
    premium = bs_price(inputs0, option_type)

    S = np.full(n_paths, S0, dtype=float)
    # Cash account starts with the premium received from selling the option.
    cash = np.full(n_paths, premium, dtype=float)
    delta_prev = np.zeros(n_paths, dtype=float)

    for step in range(n_rebalances):
        t = step * dt
        time_to_expiry = T - t

        deltas = _vectorized_bs_delta(S, K, time_to_expiry, r, sigma, q, option_type)

        trade = deltas - delta_prev
        cost = transaction_cost_bps / 10_000.0 * np.abs(trade) * S
        # Buying shares to increase hedge costs cash; selling frees cash.
        cash -= trade * S + cost
        # Cash earns/costs the risk-free rate over the interval.
        cash *= math.exp(r * dt)

        # Advance the underlying under GBM to the next rebalancing point.
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z)
        delta_prev = deltas

    # Unwind: settle the option payoff and liquidate the final hedge position.
    if option_type == OptionType.CALL:
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    final_cost = transaction_cost_bps / 10_000.0 * np.abs(delta_prev) * S
    portfolio_value = cash + delta_prev * S - payoff - final_cost

    return HedgeSimResult(pnl=portfolio_value, n_rebalances=n_rebalances)


def hedging_error_vs_frequency(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    rebalance_counts: list[int],
    option_type: OptionType = OptionType.CALL,
    n_paths: int = 5_000,
    q: float = 0.0,
    seed: int | None = None,
) -> dict[int, HedgeSimResult]:
    """Run simulate_delta_hedge across a range of rebalancing frequencies.

    Convenience wrapper for producing the headline "hedging error shrinks
    with rebalancing frequency" plot.
    """
    results = {}
    for n in rebalance_counts:
        results[n] = simulate_delta_hedge(
            S0, K, T, r, sigma, n, option_type, n_paths, q, seed=seed
        )
    return results
