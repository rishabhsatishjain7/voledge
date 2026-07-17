"""
Monte Carlo pricer for European options under geometric Brownian motion.

Simulates terminal prices directly (no need for path simulation since
European payoffs only depend on S_T under GBM):
    S_T = S0 * exp[(r - q - 0.5*sigma^2) * T + sigma * sqrt(T) * Z],  Z ~ N(0,1)

Discounted average payoff estimates the price; antithetic variates
(pairing Z with -Z) are used by default to cut estimator variance
roughly in half for the same sample budget, at negligible extra cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .black_scholes import OptionType


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo pricing result with a 95% confidence interval."""

    price: float
    std_error: float

    @property
    def ci_95(self) -> tuple[float, float]:
        half_width = 1.96 * self.std_error
        return (self.price - half_width, self.price + half_width)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
    n_paths: int = 100_000,
    q: float = 0.0,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Price a European option via Monte Carlo simulation.

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes-style inputs.
        option_type: CALL or PUT.
        n_paths: Number of simulated terminal prices (pairs, if antithetic).
        antithetic: Use antithetic variates for variance reduction.
        seed: RNG seed for reproducibility.

    Returns:
        MCResult with the price estimate and its standard error.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol_term = sigma * math.sqrt(T) if T > 0 else 0.0

    if antithetic:
        half_n = n_paths // 2
        z = rng.standard_normal(half_n)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    S_T = S * np.exp(drift + vol_term * z)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    disc = math.exp(-r * T)
    discounted = disc * payoffs

    price = float(np.mean(discounted))
    std_error = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))

    return MCResult(price=price, std_error=std_error)
