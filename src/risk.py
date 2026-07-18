"""
Risk metrics for a single option position: Value-at-Risk (VaR) and
scenario stress testing.

Two VaR methods are implemented deliberately, because they make different
assumptions and a real risk desk cross-checks one against the other
rather than trusting either alone:

    - Parametric (delta-normal): fast, closed-form, but linearizes the
      position's P&L via delta only - it misses convexity (gamma) and
      assumes P&L is normally distributed, which breaks down for options
      with large gamma/short time-to-expiry or for large market moves.
    - Monte Carlo (full revaluation): simulates the underlying forward
      under GBM over the risk horizon, then fully reprices the option at
      each simulated terminal spot (capturing gamma, theta decay, and any
      non-normality in the payoff) rather than linearizing around today's
      Greeks. Slower, but the more trustworthy of the two for anything
      with meaningful convexity over the horizon.

Stress testing separately answers a different question from VaR: not
"what's the P&L at a given confidence level under normal-market
assumptions," but "what happens to this specific position under named,
economically motivated scenarios" (a spot crash, a vol spike, a rate
move) - the kind of table a risk manager asks for directly, independent
of any distributional assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pricing.black_scholes import BSInputs, OptionType, bs_price


@dataclass(frozen=True)
class VaRResult:
    var: float          # positive number: the loss threshold at `confidence`
    cvar: float         # expected loss given that the VaR threshold is breached
    confidence: float
    horizon_days: int
    pnl_distribution: np.ndarray  # full simulated P&L array (Monte Carlo only; empty for parametric)


def parametric_var(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
    position_size: float = 1.0,
    horizon_days: int = 1,
    confidence: float = 0.95,
    q: float = 0.0,
) -> VaRResult:
    """Delta-normal VaR for a single option position.

    Approximates P&L over the horizon as delta * dS, where dS is assumed
    normally distributed with the given annualized sigma scaled to the
    horizon. This is the standard fast/parametric approach, but it misses
    gamma entirely - see module docstring and monte_carlo_var for the
    fuller alternative.

    Args:
        S, K, T, r, sigma, q: Standard option inputs (T is time to the
            option's own expiry, NOT the VaR horizon).
        option_type: CALL or PUT.
        position_size: Number of contracts/units (negative = short).
        horizon_days: VaR horizon in calendar days (e.g. 1 for daily VaR).
        confidence: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        VaRResult with `var` and `cvar` as positive loss amounts.
    """
    from scipy.stats import norm
    from .greeks import compute_greeks

    inputs = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    delta = compute_greeks(inputs, option_type).delta

    horizon_years = horizon_days / 365.0
    horizon_vol = S * sigma * math.sqrt(horizon_years)  # dollar vol of the underlying over horizon

    position_dollar_delta = position_size * delta
    pnl_std = abs(position_dollar_delta) * horizon_vol

    z = norm.ppf(confidence)
    var = z * pnl_std
    # For a normal distribution, CVaR (expected shortfall) has a closed form:
    cvar = pnl_std * norm.pdf(z) / (1 - confidence)

    return VaRResult(
        var=float(var),
        cvar=float(cvar),
        confidence=confidence,
        horizon_days=horizon_days,
        pnl_distribution=np.array([]),
    )


def monte_carlo_var(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
    position_size: float = 1.0,
    horizon_days: int = 1,
    confidence: float = 0.95,
    q: float = 0.0,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> VaRResult:
    """Full-revaluation Monte Carlo VaR for a single option position.

    Simulates the underlying forward under GBM to the VaR horizon, fully
    reprices the option (via Black-Scholes) at each simulated spot with
    the remaining time to the option's own expiry, and computes the P&L
    distribution directly - capturing gamma and any skew/kurtosis in the
    option's own payoff that the parametric method's linear approximation
    misses entirely.

    Args:
        S, K, T, r, sigma, q: Standard option inputs (T is time to the
            option's own expiry).
        option_type: CALL or PUT.
        position_size: Number of contracts/units (negative = short).
        horizon_days: VaR horizon in calendar days.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).
        n_paths: Number of simulated horizon-end spot prices.
        seed: RNG seed for reproducibility.

    Returns:
        VaRResult with `var`, `cvar`, and the full simulated P&L array.
    """
    rng = np.random.default_rng(seed)
    horizon_years = horizon_days / 365.0
    remaining_T = max(T - horizon_years, 0.0)

    today_price = bs_price(BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q), option_type)

    Z = rng.standard_normal(n_paths)
    S_horizon = S * np.exp(
        (r - q - 0.5 * sigma**2) * horizon_years + sigma * math.sqrt(horizon_years) * Z
    )

    horizon_prices = np.array(
        [bs_price(BSInputs(S=s, K=K, T=remaining_T, r=r, sigma=sigma, q=q), option_type)
         for s in S_horizon]
    )

    pnl = position_size * (horizon_prices - today_price)
    losses = -pnl  # VaR/CVaR are conventionally quoted as positive loss numbers

    var = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var

    return VaRResult(
        var=var,
        cvar=cvar,
        confidence=confidence,
        horizon_days=horizon_days,
        pnl_distribution=pnl,
    )


def stress_test(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.CALL,
    position_size: float = 1.0,
    q: float = 0.0,
    spot_shocks_pct: tuple[float, ...] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
    vol_shocks_pts: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10),
) -> pd.DataFrame:
    """Scenario stress test: reprice the position under a grid of spot
    and volatility shocks, reporting P&L relative to today's price.

    This answers a different question than VaR: not "what loss level is
    breached X% of the time," but "what is the P&L under this named,
    economically motivated scenario" - e.g. "spot down 20%, vol up 10
    points" is a specific, interpretable crisis scenario a risk manager
    can reason about directly, independent of any distributional
    assumption about how likely it is.

    Args:
        S, K, T, r, sigma, q: Standard option inputs, at today's levels.
        option_type: CALL or PUT.
        position_size: Number of contracts/units (negative = short).
        spot_shocks_pct: Relative spot moves to apply (e.g. -0.20 = -20%).
        vol_shocks_pts: Absolute vol shocks in vol points (e.g. 0.05 = +5
            volatility points, i.e. sigma + 0.05).

    Returns:
        DataFrame indexed by spot shock, columns are vol shocks, values
        are P&L in dollars relative to today's position value.
    """
    today_price = bs_price(BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q), option_type)
    today_value = position_size * today_price

    rows = []
    for spot_shock in spot_shocks_pct:
        row = {}
        shocked_S = S * (1 + spot_shock)
        for vol_shock in vol_shocks_pts:
            shocked_sigma = max(sigma + vol_shock, 1e-4)
            shocked_price = bs_price(
                BSInputs(S=shocked_S, K=K, T=T, r=r, sigma=shocked_sigma, q=q), option_type
            )
            shocked_value = position_size * shocked_price
            row[f"vol {vol_shock:+.0%}"] = shocked_value - today_value
        rows.append(row)

    index = [f"spot {s:+.0%}" for s in spot_shocks_pct]
    return pd.DataFrame(rows, index=index)
