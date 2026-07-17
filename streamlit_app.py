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

st.set_page_config(page_title="VolEdge", layout="wide")
st.title("VolEdge — Options Pricing & Hedging Engine")
st.caption(
    "Black-Scholes, binomial tree, and Monte Carlo pricers validated against "
    "each other; live implied vol smile from real options data; delta-hedging "
    "simulation. See the README for methodology and limitations."
)

RISK_FREE_RATE = 0.045  # approximate short-term T-bill rate; not fetched live

tab_smile, tab_hedge = st.tabs(["Live Vol Smile", "Delta-Hedging Simulator"])

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
