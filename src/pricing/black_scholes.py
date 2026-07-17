"""
Black-Scholes-Merton pricing for European options.

Formula (call):
    C = S0 * e^{-qT} * N(d1) - K * e^{-rT} * N(d2)

Formula (put), via put-call parity:
    P = K * e^{-rT} * N(-d2) - S0 * e^{-qT} * N(-d1)

where
    d1 = [ln(S0/K) + (r - q + 0.5*sigma^2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

Assumptions: European exercise, constant risk-free rate and volatility,
continuous dividend yield q, no transaction costs, lognormal asset dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class BSInputs:
    """Container for Black-Scholes inputs.

    Attributes:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously compounded risk-free rate (annualized).
        sigma: Annualized volatility of the underlying's returns.
        q: Continuous dividend yield (annualized). Defaults to 0.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError("Spot price S must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price K must be positive.")
        if self.T < 0:
            raise ValueError("Time to expiry T must be non-negative.")
        if self.sigma < 0:
            raise ValueError("Volatility sigma must be non-negative.")


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def d1_d2(inputs: BSInputs) -> tuple[float, float]:
    """Compute d1 and d2 for the Black-Scholes formula.

    Handles the T -> 0 edge case by returning +/- infinity appropriately
    so downstream N(d1)/N(d2) collapse to the correct intrinsic-value limits.
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q

    if T == 0 or sigma == 0:
        # Degenerate case: no time value / no randomness.
        moneyness = math.log(S / K) if S > 0 and K > 0 else 0.0
        sign = math.copysign(1.0, moneyness) if moneyness != 0 else 0.0
        return sign * math.inf, sign * math.inf

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(inputs: BSInputs, option_type: OptionType = OptionType.CALL) -> float:
    """Price a European option under Black-Scholes-Merton.

    Args:
        inputs: BSInputs bundle (S, K, T, r, sigma, q).
        option_type: OptionType.CALL or OptionType.PUT.

    Returns:
        Theoretical option price.
    """
    S, K, T, r, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.q

    if T == 0:
        # At expiry, price collapses to intrinsic value.
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = d1_d2(inputs)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == OptionType.CALL:
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Convenience wrapper: price a European call."""
    return bs_price(BSInputs(S, K, T, r, sigma, q), OptionType.CALL)


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Convenience wrapper: price a European put."""
    return bs_price(BSInputs(S, K, T, r, sigma, q), OptionType.PUT)
