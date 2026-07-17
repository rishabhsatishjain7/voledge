"""
Implied volatility back-out: given a market price, solve for the sigma
that makes Black-Scholes reproduce it.

Primary method: Newton-Raphson, using vega as the derivative. This
converges in ~5 iterations when it converges at all, but vega -> 0 for
deep ITM/OTM options makes it numerically unstable there.

Fallback: Brent's method (bisection-based, derivative-free) on a bracketed
interval. Slower but robust — used automatically whenever Newton-Raphson
fails to converge or walks outside a sane vol range.

Known limitation (see README): this solver works on whatever mid-price is
handed to it. It does not filter for stale quotes, wide bid-ask spreads,
or zero open interest — that filtering happens upstream in data.py. Noisy
low-volume strikes can therefore still produce a "converged" but
economically meaningless IV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .pricing.black_scholes import BSInputs, OptionType, bs_price
from .greeks import compute_greeks

MAX_VOL = 5.0  # 500% annualized vol ceiling; anything above is not economically sane
MIN_VOL = 1e-4


@dataclass(frozen=True)
class IVResult:
    """Result of an implied volatility solve."""

    iv: float | None
    converged: bool
    iterations: int
    method: str  # "newton" or "brent" or "failed"


def _newton_raphson(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float,
    initial_guess: float = 0.3,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> IVResult:
    sigma = initial_guess
    for i in range(1, max_iter + 1):
        inputs = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        price = bs_price(inputs, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return IVResult(iv=sigma, converged=True, iterations=i, method="newton")

        vega = compute_greeks(inputs, option_type).vega
        if vega < 1e-8:
            # Vega too small (deep ITM/OTM) to make progress reliably.
            return IVResult(iv=None, converged=False, iterations=i, method="newton")

        sigma -= diff / vega
        if sigma <= MIN_VOL or sigma >= MAX_VOL:
            return IVResult(iv=None, converged=False, iterations=i, method="newton")

    return IVResult(iv=None, converged=False, iterations=max_iter, method="newton")


def _brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float,
    lo: float = MIN_VOL,
    hi: float = MAX_VOL,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IVResult:
    def f(sigma: float) -> float:
        return bs_price(BSInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q), option_type) - market_price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # No sign change in [lo, hi]: market price is outside the range of
        # prices Black-Scholes can produce for any vol (e.g. below intrinsic
        # value, or a stale/crossed quote). No solution exists.
        return IVResult(iv=None, converged=False, iterations=0, method="failed")

    for i in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return IVResult(iv=mid, converged=True, iterations=i, method="brent")
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return IVResult(iv=(lo + hi) / 2, converged=True, iterations=max_iter, method="brent")


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = OptionType.CALL,
    q: float = 0.0,
) -> IVResult:
    """Back out implied volatility from a market option price.

    Tries Newton-Raphson first (fast); falls back to bisection/Brent
    (robust) if Newton-Raphson fails to converge.

    Args:
        market_price: Observed market price of the option.
        S, K, T, r, q: Standard Black-Scholes inputs (sigma is the unknown).
        option_type: CALL or PUT.

    Returns:
        IVResult with the solved vol (or None if no solution exists) and
        metadata about how it was obtained.
    """
    if T <= 0:
        return IVResult(iv=None, converged=False, iterations=0, method="failed")

    intrinsic = max(S - K, 0.0) if option_type == OptionType.CALL else max(K - S, 0.0)
    if market_price < intrinsic - 1e-8:
        # Below intrinsic value: not arbitrage-free, no valid IV exists.
        return IVResult(iv=None, converged=False, iterations=0, method="failed")

    result = _newton_raphson(market_price, S, K, T, r, option_type, q)
    if result.converged:
        return result

    return _brent(market_price, S, K, T, r, option_type, q)
