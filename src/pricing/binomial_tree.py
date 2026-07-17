"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Discretizes the underlying's price path over N steps with up/down factors:
    u = e^{sigma * sqrt(dt)},  d = 1/u
and risk-neutral probability:
    p = (e^{(r - q) * dt} - d) / (u - d)

Backward induction gives the option value at t=0. Supports both European
exercise (for convergence checks against Black-Scholes) and American
exercise (early-exercise optionality), since American pricing has no
closed-form solution and this is precisely where a tree earns its keep.

As N -> infinity, the CRR European price converges to the Black-Scholes
price (see tests/test_pricing.py::test_binomial_converges_to_bs).
"""

from __future__ import annotations

import math

from .black_scholes import OptionType


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: OptionType = OptionType.CALL,
    american: bool = False,
    q: float = 0.0,
) -> float:
    """Price a European or American option via a CRR binomial tree.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.
        n_steps: Number of discretization steps. Higher = more accurate,
            slower. 200-500 is typically enough for convergence to 1e-3.
        option_type: CALL or PUT.
        american: If True, allow early exercise at each node.
        q: Continuous dividend yield.

    Returns:
        Option price at t=0.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    if T == 0:
        intrinsic = (S - K) if option_type == OptionType.CALL else (K - S)
        return max(intrinsic, 0.0)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside [0, 1]; "
            "check that sigma and dt are consistent (arbitrage-free tree)."
        )

    # Terminal asset prices at each of the n_steps+1 nodes.
    asset_prices = [S * (u ** (n_steps - i)) * (d ** i) for i in range(n_steps + 1)]

    if option_type == OptionType.CALL:
        values = [max(price - K, 0.0) for price in asset_prices]
    else:
        values = [max(K - price, 0.0) for price in asset_prices]

    # Backward induction.
    for step in range(n_steps - 1, -1, -1):
        asset_prices = [S * (u ** (step - i)) * (d ** i) for i in range(step + 1)]
        values = [
            disc * (p * values[i] + (1 - p) * values[i + 1]) for i in range(step + 1)
        ]
        if american:
            if option_type == OptionType.CALL:
                intrinsic = [max(price - K, 0.0) for price in asset_prices]
            else:
                intrinsic = [max(K - price, 0.0) for price in asset_prices]
            values = [max(v, iv) for v, iv in zip(values, intrinsic)]

    return values[0]
