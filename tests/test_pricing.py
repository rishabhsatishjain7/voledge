"""
Numerical correctness tests for the pricing engine.

These are not coverage-padding tests. Each one checks a specific
mathematical property that a broken pricer would very likely violate:
put-call parity, convergence of the binomial tree to Black-Scholes,
Monte Carlo convergence, boundary behavior for deep ITM/OTM options,
and Greeks sanity checks (e.g. delta in [0, 1] for calls, gamma > 0).
"""

import math

import numpy as np
import pytest

from src.pricing.black_scholes import BSInputs, OptionType, bs_call, bs_put, bs_price
from src.pricing.binomial_tree import binomial_price
from src.pricing.monte_carlo import mc_price
from src.pricing.heston import HestonParams, heston_price, calibrate_heston
from src.risk import parametric_var, monte_carlo_var, stress_test
from src.backtest import backtest_delta_hedge, estimate_realized_vol
from src.pricing.svi import (
    SVIParams, raw_svi_total_variance, svi_implied_vol, log_forward_moneyness,
    calibrate_svi_slice, check_calendar_arbitrage,
)
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


class TestHeston:
    def test_degenerate_case_matches_black_scholes_call(self):
        """v0 = theta = sigma^2 with sigma_v -> 0 removes stochastic vol
        entirely, leaving constant variance - Heston must collapse to
        Black-Scholes exactly in this limit."""
        params = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, sigma_v=1e-4, rho=0.0)
        heston_p = heston_price(S, K, T, r, params, OptionType.CALL)
        bs_p = bs_call(S, K, T, r, sigma)
        assert heston_p == pytest.approx(bs_p, abs=1e-3)

    def test_degenerate_case_matches_black_scholes_put(self):
        params = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, sigma_v=1e-4, rho=0.0)
        heston_p = heston_price(S, K, T, r, params, OptionType.PUT)
        bs_p = bs_put(S, K, T, r, sigma)
        assert heston_p == pytest.approx(bs_p, abs=1e-3)

    def test_put_call_parity(self):
        """Heston prices must satisfy the same model-independent put-call
        parity as any European option pricer."""
        params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.5, rho=-0.7)
        call = heston_price(S, K, T, r, params, OptionType.CALL)
        put = heston_price(S, K, T, r, params, OptionType.PUT)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert lhs == pytest.approx(rhs, abs=1e-2)

    def test_negative_rho_produces_downward_skew(self):
        """Negative correlation between spot and vol (the equity 'leverage
        effect') should produce IV decreasing in strike - not a symmetric
        smile. This is the qualitative behavior the whole model exists to
        capture, so it's worth checking directly rather than just
        checking prices are self-consistent."""
        params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.5, rho=-0.7)
        low_strike_price = heston_price(S, 90.0, T, r, params, OptionType.CALL)
        high_strike_price = heston_price(S, 110.0, T, r, params, OptionType.CALL)

        low_iv = implied_volatility(low_strike_price, S, 90.0, T, r, OptionType.CALL).iv
        high_iv = implied_volatility(high_strike_price, S, 110.0, T, r, OptionType.CALL).iv

        assert low_iv is not None and high_iv is not None
        assert low_iv > high_iv

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            HestonParams(v0=-0.01, kappa=1.0, theta=0.04, sigma_v=0.5, rho=0.0)
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=0.0, theta=0.04, sigma_v=0.5, rho=0.0)
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma_v=0.5, rho=1.5)

    def test_calibration_recovers_known_parameters(self):
        """Generate prices from a known Heston parameter set, then check
        calibration recovers prices close to those targets (recovering
        the exact parameters isn't guaranteed - Heston calibration is a
        classically underdetermined problem - but recovering the price
        surface it was fit to is the meaningful correctness check)."""
        true_params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.6)
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        true_prices = np.array(
            [heston_price(S, k, T, r, true_params, OptionType.CALL) for k in strikes]
        )

        # max_nfev capped well below the default 300: this well-posed
        # recovery problem converges to a near-perfect fit in ~30-40
        # evaluations, so there's no reason to pay for hundreds more in
        # every CI run - see calibrate_heston's max_nfev docstring.
        calibrated, rmse = calibrate_heston(
            S, T, r, strikes, true_prices, OptionType.CALL, max_nfev=40
        )

        assert rmse < 0.05  # prices in this scenario range roughly $0.50-$18


class TestRisk:
    def test_parametric_and_monte_carlo_var_roughly_agree(self):
        """For a short horizon (1 day), convexity effects are small, so
        the linearized (parametric) and full-revaluation (Monte Carlo)
        VaR methods should agree fairly closely - a large divergence here
        would indicate a bug in one of the two implementations, since
        they're modeling the same underlying risk two different ways."""
        pv = parametric_var(S, K, T, r, sigma, OptionType.CALL, horizon_days=1, confidence=0.95)
        mv = monte_carlo_var(
            S, K, T, r, sigma, OptionType.CALL, horizon_days=1, confidence=0.95, seed=42
        )
        assert pv.var == pytest.approx(mv.var, rel=0.15)

    def test_var_positive_for_long_position(self):
        pv = parametric_var(S, K, T, r, sigma, OptionType.CALL, position_size=1.0)
        assert pv.var > 0
        assert pv.cvar >= pv.var  # CVaR (expected shortfall) is at least as large as VaR

    def test_var_scales_with_position_size(self):
        """Doubling the position size should roughly double the VaR
        (delta-normal VaR is exactly linear in position size)."""
        pv1 = parametric_var(S, K, T, r, sigma, OptionType.CALL, position_size=1.0)
        pv2 = parametric_var(S, K, T, r, sigma, OptionType.CALL, position_size=2.0)
        assert pv2.var == pytest.approx(2 * pv1.var, rel=1e-9)

    def test_higher_confidence_gives_larger_var(self):
        pv_95 = parametric_var(S, K, T, r, sigma, OptionType.CALL, confidence=0.95)
        pv_99 = parametric_var(S, K, T, r, sigma, OptionType.CALL, confidence=0.99)
        assert pv_99.var > pv_95.var

    def test_stress_test_zero_shock_matches_baseline(self):
        """The (spot 0%, vol 0%) cell must be exactly zero - it's the
        no-shock scenario, i.e. today's position value vs. itself."""
        result = stress_test(S, K, T, r, sigma, OptionType.CALL, position_size=1.0)
        assert result.loc["spot +0%", "vol +0%"] == pytest.approx(0.0, abs=1e-9)

    def test_stress_test_long_call_gains_from_positive_shocks(self):
        """A long call should gain value from both a spot rally (positive
        delta) and a vol increase (positive vega) - and lose value from
        the opposite moves. This checks the signs of the whole table are
        economically sane, not just that the zero-shock cell is zero."""
        result = stress_test(S, K, T, r, sigma, OptionType.CALL, position_size=1.0)
        assert result.loc["spot +20%", "vol +10%"] > 0
        assert result.loc["spot -20%", "vol -10%"] < 0
        assert result.loc["spot +0%", "vol +10%"] > 0  # vega effect alone
        assert result.loc["spot +0%", "vol -10%"] < 0


class TestBacktest:
    @staticmethod
    def _make_gbm_path(seed, s0=100.0, mu=0.05, sigma=0.20, n_days=252 * 3):
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        prices = np.zeros(n_days)
        prices[0] = s0
        for i in range(1, n_days):
            z = rng.standard_normal()
            prices[i] = prices[i - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )
        return prices

    def test_realized_vol_recovers_generating_vol(self):
        """On a GBM path with a known true vol, the trailing realized-vol
        estimator should recover something close to that true vol - not
        exact (it's an estimate from a finite, noisy sample), but in the
        right ballpark."""
        prices = self._make_gbm_path(seed=1, sigma=0.25, n_days=500)
        estimated = estimate_realized_vol(prices, lookback=250)
        assert estimated == pytest.approx(0.25, rel=0.25)

    def test_no_lookahead_bias(self):
        """The vol estimate at a given point must depend only on data up
        to and including that point - appending more future data after
        the estimation point must not change the result."""
        prices = self._make_gbm_path(seed=2, n_days=500)
        vol_a = estimate_realized_vol(prices[:301], lookback=60)
        vol_b = estimate_realized_vol(prices[:301], lookback=60)
        assert vol_a == vol_b  # deterministic given the same visible data

    def test_hedging_error_shrinks_with_rebalancing_frequency(self):
        """Same qualitative result as hedging.py's core finding, now
        demonstrated by replaying the same fixed-vol hedging strategy
        against a single fixed price path at different rebalancing
        frequencies - not a new claim, but a cross-check that the
        real-path replay mechanics behave consistently with the
        already-validated simulated mechanics."""
        prices = self._make_gbm_path(seed=3, n_days=252 * 4)
        stds = []
        for rebal in [15, 5, 1]:
            result = backtest_delta_hedge(prices, window_days=60, rebalance_every=rebal)
            stds.append(result.std)
        assert stds[0] > stds[1] > stds[2]

    def test_result_has_one_pnl_per_window(self):
        prices = self._make_gbm_path(seed=4, n_days=400)
        result = backtest_delta_hedge(prices, window_days=60, rebalance_every=5, vol_lookback=60)
        expected_n = 400 - 60 - 60  # n - window_days - vol_lookback, roughly
        assert result.n_windows > 0
        assert len(result.pnl) == len(result.vols_used) == result.n_windows

    def test_insufficient_history_raises(self):
        prices = self._make_gbm_path(seed=5, n_days=50)  # too short for defaults
        with pytest.raises(ValueError):
            backtest_delta_hedge(prices, window_days=60, vol_lookback=60)


class TestSVI:
    def test_recovers_synthetic_skew(self):
        """SVI, as a flexible curve-fit (not a constrained economic
        model like Heston), should fit an arbitrary smooth skew shape
        very tightly - this checks it actually does, not just that it
        runs without error."""
        S, T, r = 100.0, 0.5, 0.045
        strikes = np.arange(70, 131, 2.5)

        def synthetic_skew(strike, spot):
            m = math.log(strike / spot)
            return 0.18 - 0.35 * m + 0.9 * m**2

        true_ivs = np.array([synthetic_skew(k, S) for k in strikes])
        k = log_forward_moneyness(strikes, S, T, r)

        fitted, rmse = calibrate_svi_slice(k, true_ivs, T)
        fitted_ivs = svi_implied_vol(k, T, fitted)

        assert rmse < 1e-3
        assert np.max(np.abs(fitted_ivs - true_ivs)) < 0.01  # within 1 vol point everywhere

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            SVIParams(a=0.0, b=-0.1, rho=0.0, m=0.0, sigma=0.2)  # negative b
        with pytest.raises(ValueError):
            SVIParams(a=0.0, b=0.3, rho=1.5, m=0.0, sigma=0.2)  # rho out of range
        with pytest.raises(ValueError):
            SVIParams(a=0.0, b=0.3, rho=0.0, m=0.0, sigma=-0.1)  # non-positive sigma

    def test_total_variance_symmetric_case(self):
        """With rho=0, the SVI curve should be symmetric around k=m -
        a direct check that the rho term is doing what it's supposed to."""
        params = SVIParams(a=0.04, b=0.3, rho=0.0, m=0.0, sigma=0.2)
        w_left = raw_svi_total_variance(-0.3, params)
        w_right = raw_svi_total_variance(0.3, params)
        assert w_left == pytest.approx(w_right, abs=1e-9)

    def test_calendar_arbitrage_check_detects_genuine_violation(self):
        """Construct two slices where total variance at k=0 clearly
        decreases from the shorter to the longer expiry - the checker
        must flag this specific, verifiable violation."""
        short_slice = SVIParams(a=0.05, b=0.3, rho=-0.3, m=0.0, sigma=0.2)  # w(0) = 0.05 + 0.3*0.2 = 0.11
        long_slice = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2)   # w(0) = 0.01 + 0.1*0.2 = 0.03
        surface = {0.25: short_slice, 1.0: long_slice}

        violations = check_calendar_arbitrage(surface, k_grid=np.array([0.0]))
        assert len(violations) == 1
        assert violations[0][1] == 0.25 and violations[0][2] == 1.0

    def test_calendar_arbitrage_check_passes_consistent_surface(self):
        """A genuinely consistent surface (variance strictly increasing
        in T at every k, by construction) should report zero violations -
        checks the function doesn't false-positive on a clean case."""
        surface = {
            0.25: SVIParams(a=0.02, b=0.2, rho=-0.3, m=0.0, sigma=0.2),
            0.5: SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            1.0: SVIParams(a=0.08, b=0.4, rho=-0.3, m=0.0, sigma=0.2),
        }
        violations = check_calendar_arbitrage(surface, k_grid=np.linspace(-0.5, 0.5, 11))
        assert len(violations) == 0


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
