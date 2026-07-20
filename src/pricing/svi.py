"""
SVI (Gatheral, 2004) volatility surface parameterization.

Everything else in this repo treats one expiry at a time: the smile in
implied_vol.py is solved strike-by-strike for a single T, and Heston's
calibration (heston.py) fits one parameter set to one expiry's prices.
A real vol surface needs both dimensions - strike AND expiry - fit
together in a way that lets you read off a consistent IV anywhere on the
surface, not just at the strikes that happened to trade.

SVI parameterizes total implied variance w(k) = sigma_BS(k)^2 * T as a
function of log-forward-moneyness k = ln(K/F):

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Parameters:
    a:     overall level of variance
    b:     controls the angle between the left and right wings (b >= 0)
    rho:   controls the rotation/skew of the smile (-1 < rho < 1)
    m:     translates the smile horizontally (location of the minimum)
    sigma: controls the smoothness/curvature at the minimum (sigma > 0)

This module fits SVI one expiry ("slice") at a time - the standard
starting point - and separately checks whether a set of per-expiry fits
is jointly consistent (the calendar-spread, or "no butterfly across
time," condition: total variance at a given k should not decrease as T
increases, since variance is a cumulative, non-decreasing quantity
under no-arbitrage). A full arbitrage-free SVI surface (jointly
calibrated across all expiries) is a genuinely harder research-level
problem (see Gatheral & Jacquier, 2014) - this module fits slices
independently and flags calendar-arbitrage violations as a diagnostic,
rather than enforcing them as a hard constraint during calibration.
See README Limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0:
            raise ValueError("b must be non-negative.")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be strictly between -1 and 1.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")


def raw_svi_total_variance(k: np.ndarray | float, params: SVIParams) -> np.ndarray:
    """Total implied variance w(k) = IV(k)^2 * T under the raw SVI form."""
    k = np.asarray(k, dtype=float)
    return params.a + params.b * (
        params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma**2)
    )


def svi_implied_vol(k: np.ndarray | float, T: float, params: SVIParams) -> np.ndarray:
    """Black-Scholes implied vol implied by an SVI slice at log-moneyness k."""
    w = raw_svi_total_variance(k, params)
    w = np.maximum(w, 0.0)  # guard against numerical noise producing tiny negatives
    return np.sqrt(w / T)


def log_forward_moneyness(strikes: np.ndarray, S: float, T: float, r: float, q: float = 0.0) -> np.ndarray:
    """k = ln(K/F), where F = S * exp((r - q) * T) is the forward price."""
    F = S * math.exp((r - q) * T)
    return np.log(np.asarray(strikes, dtype=float) / F)


def _residuals(x: np.ndarray, k: np.ndarray, w_market: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = x
    try:
        params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    except ValueError:
        return np.full(len(k), 1e6)
    w_model = raw_svi_total_variance(k, params)
    return w_model - w_market


def calibrate_svi_slice(
    k: np.ndarray,
    implied_vols: np.ndarray,
    T: float,
    initial_guess: SVIParams | None = None,
) -> tuple[SVIParams, float]:
    """Fit an SVI slice to one expiry's market implied vols.

    Args:
        k: Log-forward-moneyness values (see log_forward_moneyness).
        implied_vols: Market Black-Scholes implied vols at each k.
        T: Time to expiry in years, for converting IV <-> total variance.
        initial_guess: Starting SVIParams. A reasonable generic default
            (centered, mild skew) is used if not provided.

    Returns:
        (fitted SVIParams, RMSE in total-variance units).
    """
    from scipy.optimize import least_squares

    w_market = (implied_vols**2) * T

    if initial_guess is None:
        atm_var = float(np.interp(0.0, k, w_market))
        initial_guess = SVIParams(a=atm_var * 0.5, b=0.3, rho=-0.3, m=0.0, sigma=0.2)

    x0 = np.array([initial_guess.a, initial_guess.b, initial_guess.rho,
                    initial_guess.m, initial_guess.sigma])
    # Bounds keep the optimizer inside the valid SVI parameter space
    # (b >= 0, |rho| < 1, sigma > 0) with generous headroom.
    lower = [-1.0, 0.0, -0.999, -2.0, 1e-4]
    upper = [2.0, 5.0, 0.999, 2.0, 2.0]

    result = least_squares(_residuals, x0, bounds=(lower, upper), args=(k, w_market), max_nfev=500)

    a, b, rho, m, sigma = result.x
    fitted = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    w_fitted = raw_svi_total_variance(k, fitted)
    rmse = float(np.sqrt(np.mean((w_fitted - w_market) ** 2)))

    return fitted, rmse


def check_calendar_arbitrage(
    surface: dict[float, SVIParams],
    k_grid: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """Check the calendar-spread no-arbitrage condition across expiries.

    Total variance w(k, T) must be non-decreasing in T at every k (since
    variance accumulates over time under no-arbitrage). Independently
    fit per-expiry SVI slices have no reason to jointly respect this -
    this function checks and reports where they don't, as a diagnostic
    rather than a hard constraint (see module docstring).

    Args:
        surface: Dict mapping T (years) to a fitted SVIParams slice for
            that expiry.
        k_grid: Log-moneyness values to check at. Defaults to a
            standard grid spanning -1 to 1.

    Returns:
        List of (k, T_earlier, T_later) tuples where total variance
        decreased from T_earlier to T_later at that k - i.e. violations.
        Empty list means no violations found on the checked grid.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 41)

    expiries = sorted(surface.keys())
    violations = []

    for k in k_grid:
        w_by_expiry = [raw_svi_total_variance(k, surface[T]) for T in expiries]
        for i in range(len(expiries) - 1):
            if w_by_expiry[i + 1] < w_by_expiry[i] - 1e-8:
                violations.append((float(k), expiries[i], expiries[i + 1]))

    return violations
