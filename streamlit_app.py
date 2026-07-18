"""
VolEdge — interactive demo.

Pick a ticker, pull its live options chain, back out the implied
volatility smile, and run a delta-hedging simulation to see how
rebalancing frequency affects hedging error.

Run locally with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data import get_available_expiries, get_option_chain, get_spot_price, years_to_expiry
from src.implied_vol import implied_volatility
from src.pricing.black_scholes import OptionType
from src.hedging import hedging_error_vs_frequency
from src.pricing.heston import calibrate_heston, heston_price
from src.risk import parametric_var, monte_carlo_var, stress_test

st.set_page_config(page_title="VolEdge", layout="wide")
st.title("VolEdge — Options Pricing & Hedging Engine")
st.caption(
    "Black-Scholes, binomial tree, Monte Carlo, and Heston (calibrated) pricers "
    "validated against each other; live implied vol smile from real options "
    "data; delta-hedging simulation; VaR and scenario stress testing. See the "
    "README for methodology and limitations."
)

RISK_FREE_RATE = 0.045  # approximate short-term T-bill rate; not fetched live

tab_smile, tab_hedge, tab_heston, tab_risk = st.tabs(
    ["Live Vol Smile", "Delta-Hedging Simulator", "Heston Calibration", "VaR & Stress Test"]
)

# ---------------------------------------------------------------------------
# Tab 1: Live implied volatility smile
# ---------------------------------------------------------------------------
with tab_smile:
    col_input, col_chart = st.columns([1, 2])

    with col_input:
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        option_side = st.radio("Option type", ["Call", "Put"], horizontal=True)

        if ticker:
            try:
                expiries = get_available_expiries(ticker)
            except Exception as e:
                st.error(f"Could not fetch expiries for '{ticker}': {e}")
                expiries = []

            if expiries:
                expiry = st.selectbox("Expiry", expiries)
                run_button = st.button("Fetch & Solve IV Smile", type="primary")
            else:
                expiry = None
                run_button = False

    with col_chart:
        if ticker and expiries and run_button:
            with st.spinner("Fetching spot price and options chain..."):
                try:
                    spot = get_spot_price(ticker)
                    calls, puts = get_option_chain(ticker, expiry)
                    df = calls if option_side == "Call" else puts
                except Exception as e:
                    st.error(f"Data fetch failed: {e}")
                    df = None

            if df is not None and not df.empty:
                T = years_to_expiry(expiry)
                opt_type = OptionType.CALL if option_side == "Call" else OptionType.PUT

                strikes, ivs = [], []
                for _, row in df.iterrows():
                    result = implied_volatility(
                        market_price=row["mid"],
                        S=spot,
                        K=row["strike"],
                        T=T,
                        r=RISK_FREE_RATE,
                        option_type=opt_type,
                    )
                    if result.converged and result.iv is not None:
                        strikes.append(row["strike"])
                        ivs.append(result.iv)

                if strikes:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(strikes, [iv * 100 for iv in ivs], marker="o", linewidth=1.5)
                    ax.axvline(spot, color="gray", linestyle="--", alpha=0.6, label=f"Spot = {spot:.2f}")
                    ax.set_xlabel("Strike")
                    ax.set_ylabel("Implied Volatility (%)")
                    ax.set_title(f"{ticker} {option_side} IV Smile — Expiry {expiry}")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    st.caption(
                        f"Spot: {spot:.2f}  |  {len(strikes)} strikes converged "
                        f"out of {len(df)} after liquidity filtering."
                    )
                else:
                    st.warning("No strikes converged to a valid implied vol after filtering.")
            elif df is not None:
                st.warning(
                    "No strikes passed the liquidity filters (min volume / open "
                    "interest / max spread). Try a different expiry or ticker."
                )
        else:
            st.info("Enter a ticker, choose an expiry, and click Fetch & Solve IV Smile.")

# ---------------------------------------------------------------------------
# Tab 2: Delta-hedging simulator
# ---------------------------------------------------------------------------
with tab_hedge:
    st.write(
        "Simulates selling a European option and dynamically delta-hedging it "
        "under GBM. Shows how hedging error (P&L variance at expiry) shrinks "
        "as rebalancing frequency increases — the residual error from discrete "
        "vs. continuous hedging."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        S0 = st.number_input("Spot (S0)", value=100.0, min_value=1.0)
        K = st.number_input("Strike (K)", value=100.0, min_value=1.0)
    with col_b:
        T = st.number_input("Time to expiry (years)", value=1.0, min_value=0.05, max_value=5.0)
        sigma = st.number_input("Volatility (sigma)", value=0.20, min_value=0.01, max_value=2.0)
    with col_c:
        r = st.number_input("Risk-free rate", value=0.05, min_value=0.0, max_value=0.25)
        opt_side = st.radio("Type", ["Call", "Put"], horizontal=True, key="hedge_type")

    n_paths = st.slider("Number of simulated paths", 500, 10_000, 3000, step=500)
    rebalance_options = st.multiselect(
        "Rebalancing frequencies (per year) to compare",
        [4, 12, 26, 52, 126, 252],
        default=[4, 12, 52, 252],
    )

    if st.button("Run Hedging Simulation", type="primary") and rebalance_options:
        opt_type = OptionType.CALL if opt_side == "Call" else OptionType.PUT
        with st.spinner("Simulating hedged paths..."):
            results = hedging_error_vs_frequency(
                S0, K, T, r, sigma,
                rebalance_counts=sorted(rebalance_options),
                option_type=opt_type,
                n_paths=n_paths,
                seed=42,
            )

        freqs = sorted(results.keys())
        stds = [results[n].std for n in freqs]
        means = [results[n].mean for n in freqs]

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            ax1.plot(freqs, stds, marker="o", color="crimson")
            ax1.set_xlabel("Rebalances per year")
            ax1.set_ylabel("Hedging P&L Std. Dev.")
            ax1.set_title("Hedging Error vs. Rebalancing Frequency")
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            ax2.hist(results[freqs[-1]].pnl, bins=40, alpha=0.75, color="steelblue")
            ax2.axvline(0, color="black", linestyle="--", alpha=0.6)
            ax2.set_xlabel("Hedging P&L")
            ax2.set_ylabel("Frequency")
            ax2.set_title(f"P&L Distribution at {freqs[-1]} rebalances/yr")
            st.pyplot(fig2)

        st.dataframe(
            {
                "Rebalances/yr": freqs,
                "Mean P&L": [f"{m:+.4f}" for m in means],
                "Std Dev (hedging error)": [f"{s:.4f}" for s in stds],
            }
        )

# ---------------------------------------------------------------------------
# Tab 3: Heston calibration to a live options chain
# ---------------------------------------------------------------------------
MAX_CALIBRATION_STRIKES = 10  # keeps calibration responsive - see caption below
CALIBRATION_MAX_NFEV = 25     # optimizer budget; ~3-8s in practice with 10 strikes
MONEYNESS_BAND = (0.7, 1.4)   # restrict to this range of strike/spot before subsampling


def _filter_and_subsample(df, spot: float, max_n: int):
    """Restrict to a sensible moneyness band, then pick up to max_n rows
    evenly spaced across the remaining strike range.

    Two reasons to restrict the moneyness band first: deep ITM/OTM
    strikes contribute very little calibration signal (their prices are
    dominated by intrinsic value or are near-zero) while adding noise and
    hurting the optimizer's conditioning - and Heston pricing here uses
    numerical integration rather than a closed form, so fitting against
    every available strike in a live chain (which can span 50-100+
    strikes across a very wide moneyness range) would make the app feel
    unresponsive.
    """
    df = df[(df["strike"] >= spot * MONEYNESS_BAND[0]) & (df["strike"] <= spot * MONEYNESS_BAND[1])]
    df = df.sort_values("strike").reset_index(drop=True)
    if len(df) <= max_n:
        return df
    idx = sorted(set(np.linspace(0, len(df) - 1, max_n).round().astype(int)))
    return df.iloc[idx].reset_index(drop=True)


with tab_heston:
    st.write(
        "Calibrates the Heston stochastic volatility model directly to a live "
        "options chain, then compares the model-implied smile against the "
        "market's own implied vols - a direct check of how well a 5-parameter "
        "stochastic vol model can match what the market is actually pricing."
    )

    col_input, col_chart = st.columns([1, 2])

    with col_input:
        heston_ticker = st.text_input("Ticker", value="AAPL", key="heston_ticker").strip().upper()

        if heston_ticker:
            try:
                heston_expiries = get_available_expiries(heston_ticker)
            except Exception as e:
                st.error(f"Could not fetch expiries for '{heston_ticker}': {e}")
                heston_expiries = []

            if heston_expiries:
                heston_expiry = st.selectbox("Expiry", heston_expiries, key="heston_expiry")
                calibrate_button = st.button("Fetch Chain & Calibrate Heston", type="primary")
            else:
                heston_expiry = None
                calibrate_button = False

        st.caption(
            f"Calibration restricts to strikes within {MONEYNESS_BAND[0]:.0%}-"
            f"{MONEYNESS_BAND[1]:.0%} of spot, then subsamples up to "
            f"{MAX_CALIBRATION_STRIKES} evenly across that range - deep tail "
            "strikes add little calibration signal but hurt both speed and "
            "the optimizer's conditioning, since Heston pricing here uses "
            "numerical integration rather than a closed form. Takes roughly "
            "5-10 seconds."
        )

    with col_chart:
        if heston_ticker and heston_expiries and calibrate_button:
            with st.spinner("Fetching chain and calibrating Heston (this takes a few seconds)..."):
                calibrated = None
                try:
                    heston_spot = get_spot_price(heston_ticker)
                    heston_calls, _ = get_option_chain(heston_ticker, heston_expiry)
                except Exception as e:
                    st.error(f"Data fetch failed: {e}")
                    heston_calls = None

                if heston_calls is not None and not heston_calls.empty:
                    heston_T = years_to_expiry(heston_expiry)
                    sampled = _filter_and_subsample(heston_calls, heston_spot, MAX_CALIBRATION_STRIKES)
                    strikes_arr = sampled["strike"].to_numpy(dtype=float)
                    prices_arr = sampled["mid"].to_numpy(dtype=float)

                    if len(strikes_arr) < 4:
                        st.warning(
                            f"Only {len(strikes_arr)} strikes fell within the "
                            f"{MONEYNESS_BAND[0]:.0%}-{MONEYNESS_BAND[1]:.0%} "
                            "moneyness band after liquidity filtering - too few "
                            "to calibrate reliably. Try a different expiry."
                        )
                        calibrated = None
                    else:
                        try:
                            calibrated, rmse = calibrate_heston(
                                S=heston_spot, T=heston_T, r=RISK_FREE_RATE,
                                strikes=strikes_arr, market_prices=prices_arr,
                                option_type=OptionType.CALL,
                                max_nfev=CALIBRATION_MAX_NFEV,
                            )
                        except Exception as e:
                            st.error(f"Calibration failed: {e}")
                            calibrated = None

            if heston_calls is not None and heston_calls.empty:
                st.warning(
                    "No strikes passed the liquidity filters. Try a different "
                    "expiry or ticker."
                )
            elif calibrated is not None:
                market_ivs, heston_ivs, plot_strikes = [], [], []
                for k, market_price in zip(strikes_arr, prices_arr):
                    mkt_result = implied_volatility(
                        market_price, heston_spot, k, heston_T, RISK_FREE_RATE, OptionType.CALL
                    )
                    model_price = heston_price(
                        heston_spot, k, heston_T, RISK_FREE_RATE, calibrated, OptionType.CALL
                    )
                    model_result = implied_volatility(
                        model_price, heston_spot, k, heston_T, RISK_FREE_RATE, OptionType.CALL
                    )
                    if mkt_result.converged and model_result.converged:
                        plot_strikes.append(k)
                        market_ivs.append(mkt_result.iv * 100)
                        heston_ivs.append(model_result.iv * 100)

                if plot_strikes:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(plot_strikes, market_ivs, marker="o", label="Market IV")
                    ax.plot(plot_strikes, heston_ivs, marker="s", label="Heston-implied IV (calibrated)")
                    ax.axvline(heston_spot, color="gray", linestyle=":", alpha=0.6, label=f"Spot = {heston_spot:.2f}")
                    ax.set_xlabel("Strike")
                    ax.set_ylabel("Implied Volatility (%)")
                    ax.set_title(f"{heston_ticker} Heston Calibration — Expiry {heston_expiry}")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                    st.caption(
                        f"Calibration RMSE: {rmse:.4f} (price units) across "
                        f"{len(strikes_arr)} sampled strikes."
                    )
                    st.write("**Calibrated parameters:**")
                    st.dataframe(
                        {
                            "Parameter": ["v0 (initial variance)", "kappa (mean-reversion speed)",
                                          "theta (long-run variance)", "sigma_v (vol of vol)",
                                          "rho (spot-vol correlation)"],
                            "Value": [f"{calibrated.v0:.4f}", f"{calibrated.kappa:.4f}",
                                      f"{calibrated.theta:.4f}", f"{calibrated.sigma_v:.4f}",
                                      f"{calibrated.rho:.4f}"],
                        },
                        hide_index=True,
                    )
                else:
                    st.warning("No strikes converged to a comparable IV after calibration.")
        else:
            st.info("Enter a ticker, choose an expiry, and click Fetch Chain & Calibrate Heston.")

# ---------------------------------------------------------------------------
# Tab 4: VaR and scenario stress testing
# ---------------------------------------------------------------------------
with tab_risk:
    st.write(
        "Computes 1-day Value-at-Risk two ways - delta-normal (fast, "
        "linearizes P&L via delta) and Monte Carlo full revaluation (slower, "
        "captures gamma) - compared directly, plus a scenario stress-test "
        "grid showing P&L under named spot/vol shocks."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        risk_S = st.number_input("Spot (S)", value=100.0, min_value=1.0, key="risk_S")
        risk_K = st.number_input("Strike (K)", value=100.0, min_value=1.0, key="risk_K")
    with col_b:
        risk_T = st.number_input("Time to expiry (years)", value=0.5, min_value=0.05, max_value=5.0, key="risk_T")
        risk_sigma = st.number_input("Volatility (sigma)", value=0.22, min_value=0.01, max_value=2.0, key="risk_sigma")
    with col_c:
        risk_r = st.number_input("Risk-free rate", value=0.05, min_value=0.0, max_value=0.25, key="risk_r")
        risk_side = st.radio("Type", ["Call", "Put"], horizontal=True, key="risk_type")

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        position_size = st.number_input(
            "Position size (contracts, negative = short)", value=100.0, key="risk_position"
        )
    with col_e:
        horizon_days = st.selectbox("VaR horizon (days)", [1, 5, 10], index=0)
    with col_f:
        confidence = st.selectbox("Confidence level", [0.95, 0.99], index=0)

    if st.button("Compute VaR & Stress Test", type="primary"):
        risk_opt_type = OptionType.CALL if risk_side == "Call" else OptionType.PUT

        with st.spinner("Computing VaR and stress scenarios..."):
            pvar = parametric_var(
                risk_S, risk_K, risk_T, risk_r, risk_sigma, risk_opt_type,
                position_size=position_size, horizon_days=horizon_days, confidence=confidence,
            )
            mvar = monte_carlo_var(
                risk_S, risk_K, risk_T, risk_r, risk_sigma, risk_opt_type,
                position_size=position_size, horizon_days=horizon_days, confidence=confidence,
                n_paths=20_000, seed=42,
            )
            stress_df = stress_test(
                risk_S, risk_K, risk_T, risk_r, risk_sigma, risk_opt_type,
                position_size=position_size,
            )

        st.write(f"**{horizon_days}-day {confidence:.0%} VaR** for a {position_size:+.0f}-contract position:")
        st.dataframe(
            {
                "Method": ["Parametric (delta-normal)", "Monte Carlo (full revaluation)"],
                "VaR": [f"{pvar.var:.2f}", f"{mvar.var:.2f}"],
                "CVaR": [f"{pvar.cvar:.2f}", f"{mvar.cvar:.2f}"],
            },
            hide_index=True,
        )
        relative_diff = abs(pvar.var - mvar.var) / mvar.var if mvar.var > 0 else 0
        st.caption(
            f"Relative difference between methods: {relative_diff:.1%} — a large "
            "gap here would suggest meaningful gamma/convexity effects the "
            "parametric method's linear approximation is missing."
        )

        st.write("**Scenario stress test — P&L relative to today's position value ($):**")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(stress_df.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(stress_df.columns)))
        ax.set_xticklabels(stress_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(stress_df.index)))
        ax.set_yticklabels(stress_df.index)
        for i in range(len(stress_df.index)):
            for j in range(len(stress_df.columns)):
                ax.text(j, i, f"{stress_df.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="P&L ($)")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Set the position parameters above and click Compute VaR & Stress Test.")
