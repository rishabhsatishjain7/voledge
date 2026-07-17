"""
Numerical correctness tests for the pricing engine.

These are not coverage-padding tests. Each one checks a specific
mathematical property that a broken pricer would very likely violate:
put-call parity, convergence of the binomial tree to Black-Scholes,
Monte Carlo convergence, boundary behavior for deep ITM/OTM options,
and Greeks sanity checks (e.g. delta in [0, 1] for calls, gamma > 0).
"""

import math

import pytest

from src.pricing.black_scholes import BSInputs, OptionType, bs_call, bs_put, bs_price
from src.pricing.binomial_tree import binomial_price
from src.pricing.monte_carlo import mc_price
from src.greeks import compute_greeks
from src.implied_vol import implied_volatility

# Common test scenario: a fairly standard at-the-money-ish option.
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBlackScholes:
    def test_put_call_parity(self):
        """C - P = S*e^{-qT} - K*e^{-rT} must hold exactly (closed-form)."""
        call = bs_call(S, K, T, r, sigma)
        put = bs_put(S, K, T, r, sigma)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-8)

    def test_put_call_parity_with_dividends(self):
        q = 0.02
        call = bs_call(S, K, T, r, sigma, q)
        put = bs_put(S, K, T, r, sigma, q)
        lhs = call - put
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-8)

    def test_deep_itm_call_converges_to_intrinsic(self):
        """Deep ITM call: time value vanishes, price -> S - K*e^{-rT}."""
        deep_itm_call = bs_call(S=1000.0, K=100.0, T=T, r=r, sigma=sigma)
        forward_intrinsic = 1000.0 - 100.0 * math.exp(-r * T)
        assert deep_itm_call == pytest.approx(forward_intrinsic, rel=1e-4)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be worth close to nothing."""
        deep_otm_call = bs_call(S=100.0, K=1000.0, T=T, r=r, sigma=sigma)
        assert deep_otm_call == pytest.approx(0.0, abs=1e-6)

    def test_zero_vol_collapses_to_forward_intrinsic(self):
        """sigma -> 0 removes randomness; price = discounted intrinsic."""
        call = bs_call(S=110.0, K=100.0, T=T, r=r, sigma=1e-6)
        expected = 110.0 - 100.0 * math.exp(-r * T)
        assert call == pytest.approx(expected, rel=1e-3)

    def test_at_expiry_equals_intrinsic(self):
        """T=0: price must equal exact intrinsic value, no time value."""
        assert bs_call(S=110.0, K=100.0, T=0.0, r=r, sigma=sigma) == pytest.approx(10.0)
        assert bs_put(S=90.0, K=100.0, T=0.0, r=r, sigma=sigma) == pytest.approx(10.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            BSInputs(S=-10, K=100, T=1, r=0.05, sigma=0.2)
        with pytest.raises(ValueError):
            BSInputs(S=100, K=100, T=-1, r=0.05, sigma=0.2)


class TestBinomialTree:
    def test_converges_to_black_scholes_call(self):
        """European binomial price -> BS price as n_steps grows."""
        bs = bs_call(S, K, T, r, sigma)
        tree_coarse = binomial_price(S, K, T, r, sigma, n_steps=10, option_type=OptionType.CALL)
        tree_fine = binomial_price(S, K, T, r, sigma, n_steps=500, option_type=OptionType.CALL)

        error_coarse = abs(tree_coarse - bs)
        error_fine = abs(tree_fine - bs)

        assert error_fine < error_coarse
        assert error_fine < 1e-2

    def test_converges_to_black_scholes_put(self):
        bs = bs_put(S, K, T, r, sigma)
        tree_fine = binomial_price(S, K, T, r, sigma, n_steps=500, option_type=OptionType.PUT)
        assert tree_fine == pytest.approx(bs, abs=1e-2)

    def test_american_call_no_dividends_equals_european(self):
        """With no dividends, early exercise of an American call is never
        optimal, so American == European. This is a standard no-arbitrage
        result and a good check that the American branch isn't buggy."""
        european = binomial_price(S, K, T, r, sigma, n_steps=200, american=False)
        american = binomial_price(S, K, T, r, sigma, n_steps=200, american=True)
        assert american == pytest.approx(european, abs=1e-6)

    def test_american_put_worth_at_least_european_put(self):
        """Early-exercise optionality means American put >= European put."""
        european = binomial_price(
            S, K, T, r, sigma, n_steps=200, option_type=OptionType.PUT, american=False
        )
        american = binomial_price(
            S, K, T, r, sigma, n_steps=200, option_type=OptionType.PUT, american=True
        )
        assert american >= european - 1e-9


class TestMonteCarlo:
    def test_converges_to_black_scholes(self):
        bs = bs_call(S, K, T, r, sigma)
        result = mc_price(S, K, T, r, sigma, n_paths=200_000, seed=42)
        assert abs(result.price - bs) < 3 * result.std_error

    def test_confidence_interval_contains_bs_price(self):
        bs = bs_put(S, K, T, r, sigma)
        result = mc_price(S, K, T, r, sigma, option_type=OptionType.PUT, n_paths=200_000, seed=7)
        lo, hi = result.ci_95
        assert lo <= bs <= hi


class TestGreeks:
    def test_call_delta_in_zero_one(self):
        inputs = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        greeks = compute_greeks(inputs, OptionType.CALL)
        assert 0.0 <= greeks.delta <= 1.0

    def test_put_delta_in_neg_one_zero(self):
        inputs = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        greeks = compute_greeks(inputs, OptionType.PUT)
        assert -1.0 <= greeks.delta <= 0.0

    def test_gamma_positive_and_equal_for_call_and_put(self):
        """Gamma is identical for calls and puts at the same strike (a
        direct consequence of put-call parity: d(Delta_call)/dS =
        d(Delta_put)/dS since they differ by a constant)."""
        inputs = BSInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        call_greeks = compute_greeks(inputs, OptionType.CALL)
        put_greeks = compute_greeks(inputs, OptionType.PUT)
        assert call_greeks.gamma > 0
        assert call_greeks.gamma == pytest.approx(put_greeks.gamma, rel=1e-9)

    def test_delta_matches_finite_difference(self):
        """Delta should match a central finite-difference bump of price
        w.r.t. spot, independent of the closed-form derivation."""
        h = 0.01
        price_up = bs_price(BSInputs(S=S + h, K=K, T=T, r=r, sigma=sigma), OptionType.CALL)
        price_down = bs_price(BSInputs(S=S - h, K=K, T=T, r=r, sigma=sigma), OptionType.CALL)
        fd_delta = (price_up - price_down) / (2 * h)

        analytical_delta = compute_greeks(
            BSInputs(S=S, K=K, T=T, r=r, sigma=sigma), OptionType.CALL
        ).delta

        assert analytical_delta == pytest.approx(fd_delta, abs=1e-4)


class TestImpliedVol:
    def test_recovers_known_vol(self):
        """Round-trip: price at a known sigma, then solve for it back out."""
        true_sigma = 0.25
        price = bs_call(S, K, T, r, true_sigma)
        result = implied_volatility(price, S, K, T, r, OptionType.CALL)
        assert result.converged
        assert result.iv == pytest.approx(true_sigma, abs=1e-4)

    def test_below_intrinsic_returns_no_solution(self):
        """A price below intrinsic value is not arbitrage-free; no IV exists."""
        result = implied_volatility(
            market_price=5.0, S=150.0, K=100.0, T=T, r=r, option_type=OptionType.CALL
        )
        assert result.iv is None
        assert not result.converged
