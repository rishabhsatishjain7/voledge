"""
Closed-form Black-Scholes Greeks.

    Delta (call) = e^{-qT} * N(d1)
    Delta (put)  = e^{-qT} * (N(d1) - 1)
    Gamma        = e^{-qT} * phi(d1) / (S * sigma * sqrt(T))          [same for call/put]
    Vega         = S * e^{-qT} * phi(d1) * sqrt(T)                     [same for call/put, per 1.00 vol]
    Theta (call) = -S*e^{-qT}*phi(d1)*sigma / (2*sqrt(T)) - r*K*e^{-rT}*N(d2) + q*S*e^{-qT}*N(d1)
    Theta (put)  = -S*e^{-qT}*phi(d1)*sigma / (2*sqrt(T)) + r*K*e^{-rT}*N(-d2) - q*S*e^{-qT}*N(-d1)
    Rho (call)   = K*T*e^{-rT}*N(d2)
    Rho (put)    = -K*T*e^{-rT}*N(-d2)

Theta is expressed per year; divide by 365 for a per-calendar-day figure.
Vega and Rho are expressed per unit (100%) move in vol/rate; divide by 100
for the conventional "per 1% move" quoting convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pricing.black_scholes import BSInputs, OptionType, norm_cdf, norm_pdf, d1_d2


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def compute_greeks(inputs: BSInputs, option_type: OptionType = OptionType.CALL) -> Greeks:
    """Compute the full set of Black-Scholes Greeks for a European option.

    Args:
        inputs: BSInputs bundle (S, K, T, r, sigma, q).
        option_type: CALL or PUT.

    Returns:
        Greeks dataclass with delta, gamma, vega, theta, rho.
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q

    if T == 0 or sigma == 0:
        # At/near expiry, gamma/vega/theta blow up or vanish depending on
        # moneyness; return the well-defined limiting values.
        itm = S > K if option_type == OptionType.CALL else S < K
        delta = float(itm) if option_type == OptionType.CALL else -float(S < K)
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = d1_d2(inputs)
    sqrt_T = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T

    if option_type == OptionType.CALL:
        delta = disc_q * norm_cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * disc_r * norm_cdf(d2)
            + q * S * disc_q * norm_cdf(d1)
        )
        rho = K * T * disc_r * norm_cdf(d2)
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * disc_r * norm_cdf(-d2)
            - q * S * disc_q * norm_cdf(-d1)
        )
        rho = -K * T * disc_r * norm_cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
