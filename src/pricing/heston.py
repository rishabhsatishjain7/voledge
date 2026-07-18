"""
Heston (1993) stochastic volatility model.

Motivation: every pricer elsewhere in this repo (Black-Scholes, the CRR
tree, Monte Carlo under GBM) assumes a single constant volatility. The
implied vol smile produced by implied_vol.py is direct empirical evidence
that assumption is wrong - if it were right, every strike would back out
the same sigma, and it doesn't. Heston fixes this by letting variance
itself follow a mean-reverting stochastic process, correlated with the
asset's returns (the correlation is what produces skew, not just smile
curvature).

SDEs under the risk-neutral measure:
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW_t^v
    corr(dW^S, dW^v) = rho

Parameters:
    v0:      initial variance
    kappa:   mean-reversion speed of variance
    theta:   long-run mean variance
    sigma_v: volatility of variance ("vol of vol")
    rho:     correlation between asset and variance shocks (typically
             negative for equities - the leverage effect: price drops
             coincide with vol spikes, which is what produces downward-
             sloping skew rather than a symmetric smile)

Pricing follows the original Heston (1993) semi-closed-form: the price
is expressed via two probabilities P1, P2 (analogous to Black-Scholes'
N(d1), N(d2)), each recovered from its own characteristic function via
numerical integration (Gil-Pelaez inversion). Uses the "Little Trap"
formulation (Albrecher, Mayer, Schoutens & Tistaert, 2007), which avoids
the branch-cut discontinuities that plague the textbook formula for
longer maturities or certain parameter combinations.

Feller condition (2*kappa*theta > sigma_v^2) is not enforced here: it
guarantees variance stays strictly positive in continuous time, but
real calibrated parameter sets frequently violate it, and this pricing
approach (unlike an Euler-discretized SDE simulation) doesn't require it
to produce a valid price. See README Limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .black_scholes import OptionType


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0 or self.theta < 0 or self.sigma_v < 0:
            raise ValueError("v0, theta, and sigma_v must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa (mean-reversion speed) must be positive.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")


def _heston_char_func(
    u: np.ndarray | float,
    x0: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    j: int,
) -> np.ndarray:
    """Characteristic function phi_j(u) for j in {1, 2}, "Little Trap" form.

    j=1 corresponds to the characteristic function under the "stock"
    measure (used for P1), j=2 under the risk-neutral money-market
    measure (used for P2) - mirroring the two probabilities in the
    Black-Scholes N(d1)/N(d2) decomposition.
    """
    i = 1j
    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )

    if j == 1:
        b = kappa - rho * sigma_v
        u_j = 0.5
    else:
        b = kappa
        u_j = -0.5

    a = kappa * theta
    d = np.sqrt((rho * sigma_v * i * u - b) ** 2 - sigma_v**2 * (2 * u_j * i * u - u**2))
    g = (b - rho * sigma_v * i * u - d) / (b - rho * sigma_v * i * u + d)

    exp_dT = np.exp(-d * T)
    C = (r - q) * i * u * T + (a / sigma_v**2) * (
        (b - rho * sigma_v * i * u - d) * T - 2.0 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - rho * sigma_v * i * u - d) / sigma_v**2) * ((1 - exp_dT) / (1 - g * exp_dT))

    return np.exp(C + D * v0 + i * u * x0)


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: OptionType = OptionType.CALL,
    q: float = 0.0,
    upper_limit: float = 100.0,
) -> float:
    """Price a European option under Heston via Gil-Pelaez inversion.

    Args:
        S, K, T, r, q: Standard option pricing inputs.
        params: HestonParams (v0, kappa, theta, sigma_v, rho).
        option_type: CALL or PUT.
        upper_limit: Truncation point for the semi-infinite integral.
            100 is generous for typical equity-scale parameters; the
            integrand decays quickly, but very high vol-of-vol can need
            a larger value - see calibrate_heston's residual check.

    Returns:
        Theoretical option price.
    """
    if T <= 0:
        intrinsic = (S - K) if option_type == OptionType.CALL else (K - S)
        return max(intrinsic, 0.0)

    x0 = math.log(S)
    lnK = math.log(K)

    def integrand(u: float, j: int) -> float:
        phi = _heston_char_func(u, x0, T, r, q, params, j)
        val = np.exp(-1j * u * lnK) * phi / (1j * u)
        return float(val.real)

    P1 = 0.5 + (1.0 / math.pi) * quad(integrand, 1e-10, upper_limit, args=(1,), limit=200)[0]
    P2 = 0.5 + (1.0 / math.pi) * quad(integrand, 1e-10, upper_limit, args=(2,), limit=200)[0]

    call = S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

    if option_type == OptionType.CALL:
        return max(call, 0.0)

    # Put via put-call parity - cheaper and more numerically stable than
    # re-deriving a separate put integral.
    put = call - S * math.exp(-q * T) + K * math.exp(-r * T)
    return max(put, 0.0)


def _residuals(
    x: np.ndarray,
    S: float,
    T: float,
    r: float,
    q: float,
    strikes: np.ndarray,
    market_prices: np.ndarray,
    option_type: OptionType,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted (model - market) residuals for least-squares calibration."""
    v0, kappa, theta, sigma_v, rho = x
    try:
        params = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)
    except ValueError:
        return np.full(len(strikes), 1e6)

    model_prices = np.array(
        [heston_price(S, k, T, r, params, option_type, q) for k in strikes]
    )
    return weights * (model_prices - market_prices)


def calibrate_heston(
    S: float,
    T: float,
    r: float,
    strikes: np.ndarray,
    market_prices: np.ndarray,
    option_type: OptionType = OptionType.CALL,
    q: float = 0.0,
    initial_guess: HestonParams | None = None,
    max_nfev: int = 300,
) -> tuple[HestonParams, float]:
    """Calibrate Heston parameters to a set of market option prices.

    Uses scipy's bounded least-squares (Trust Region Reflective) to
    minimize squared pricing error across strikes, weighting each
    residual by 1/price so that cheap far-OTM options - which would
    otherwise contribute negligible absolute error and get ignored by
    the optimizer - still pull on the fit proportionally to their
    relative price.

    Args:
        S, T, r, q: Spot, time to expiry, risk-free rate, dividend yield -
            fixed (not calibrated) for this single-expiry fit.
        strikes: Array of strikes.
        market_prices: Corresponding market mid-prices.
        option_type: CALL or PUT (all strikes assumed same type here).
        initial_guess: Starting HestonParams. Defaults to a generic
            equity-like starting point (mild vol-of-vol, negative skew)
            if not provided.
        max_nfev: Maximum residual-function evaluations for the
            optimizer. Each numerical-Jacobian iteration costs roughly
            (n_params + 1) evaluations, so this scales calibration time
            directly - lower it for interactive/live-demo use (~20-30
            gives a fit in a few seconds), raise it for offline/notebook
            use where accuracy matters more than latency.

    Returns:
        (calibrated HestonParams, final RMSE in price units).
    """
    from scipy.optimize import least_squares

    if initial_guess is None:
        initial_guess = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.4, rho=-0.6)

    x0 = np.array(
        [initial_guess.v0, initial_guess.kappa, initial_guess.theta,
         initial_guess.sigma_v, initial_guess.rho]
    )
    # Bounds: v0/theta strictly positive, kappa > 0 (mean reversion must
    # pull back), sigma_v > 0, rho strictly inside (-1, 1) for numerical
    # stability of the characteristic function near the endpoints.
    lower = [1e-4, 1e-3, 1e-4, 1e-3, -0.999]
    upper = [4.0, 20.0, 4.0, 5.0, 0.999]

    weights = 1.0 / np.maximum(market_prices, 1e-3)

    result = least_squares(
        _residuals,
        x0,
        bounds=(lower, upper),
        args=(S, T, r, q, strikes, market_prices, option_type, weights),
        max_nfev=max_nfev,
    )

    v0, kappa, theta, sigma_v, rho = result.x
    calibrated = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)

    model_prices = np.array(
        [heston_price(S, k, T, r, calibrated, option_type, q) for k in strikes]
    )
    rmse = float(np.sqrt(np.mean((model_prices - market_prices) ** 2)))

    return calibrated, rmse
