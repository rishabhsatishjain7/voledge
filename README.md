# VolEdge

A European options pricing and hedging engine: Black-Scholes, a CRR binomial
tree, and Monte Carlo pricers cross-validated against each other; an implied
volatility solver with a robust fallback for low-vega strikes; and a
delta-hedging simulation that quantifies discretization error as a function
of rebalancing frequency.

**Headline result:** delta-hedging error (P&L std. dev. at expiry) falls
monotonically as rebalancing frequency increases, from **3.26** at 4
rebalances/year down to **0.43** at 252 (daily) — while mean P&L stays flat
near zero throughout, confirming the simulation captures a variance effect,
not a directional edge.

![Hedging error vs rebalancing frequency](assets/hedging_error.png)

**Live demo:** [Streamlit app link — pick a ticker, see the live IV smile, run the hedging simulator]

---

## Methodology

- **Black-Scholes** (`src/pricing/black_scholes.py`) is the baseline
  closed-form European pricer, implemented from the formula directly
  (no external stats library — the normal CDF is built on `math.erf`).
- **Binomial tree** (`src/pricing/binomial_tree.py`) is a CRR tree used two
  ways: as a convergence check against Black-Scholes (see plot below), and
  as an American-exercise pricer, since American options have no
  closed-form solution and this is exactly where a tree earns its keep.
- **Monte Carlo** (`src/pricing/monte_carlo.py`) simulates terminal prices
  directly under GBM (European payoffs only depend on S_T, so no need to
  simulate full paths), with antithetic variates for variance reduction.
- **Implied volatility** (`src/implied_vol.py`) inverts Black-Scholes via
  Newton-Raphson (fast, ~5 iterations when it converges), falling back to
  bisection/Brent for low-vega deep ITM/OTM strikes where Newton-Raphson's
  derivative-based updates become unstable.
- **Delta hedging** (`src/hedging.py`) simulates selling an option and
  dynamically rehedging its delta exposure under GBM, financing the cash
  difference at the risk-free rate, to measure how discrete rebalancing
  error compares to the idealized continuous-time hedge.

All three pricers agree closely on a plain-vanilla ATM call
(S=K=100, T=1, r=5%, σ=20%): Black-Scholes **10.4506**, binomial tree
(500 steps) **10.4466**, Monte Carlo (200k paths) **10.4763 ± 0.0649**
(95% CI) — the closed-form price sits inside the MC confidence interval,
and the tree converges to it as step count grows.

## Results

### Binomial tree convergence to Black-Scholes

As tree depth increases, the CRR price converges to the closed-form
Black-Scholes price at the expected rate:

![Binomial convergence](assets/convergence.png)

### Implied volatility smile

The IV solver recovers a known parametric skew (used here as a
reproducible, offline stand-in for a live chain — see Limitations) across
25 strikes with a max absolute error of **2e-6**, including the low-vega
tails that force the Brent fallback:

![Vol smile recovery](assets/vol_smile.png)

### Delta-hedging error vs. rebalancing frequency

| Rebalances/yr | Std. dev. (hedging error) | Mean P&L |
|---:|---:|---:|
| 4   | 3.2599 | -0.0480 |
| 12  | 1.9353 | -0.0390 |
| 26  | 1.3500 | -0.0183 |
| 52  | 0.9408 | -0.0012 |
| 126 | 0.6234 | -0.0027 |
| 252 | 0.4318 | +0.0060 |

## Repo structure

```
voledge/
├── README.md
├── src/
│   ├── pricing/
│   │   ├── black_scholes.py
│   │   ├── binomial_tree.py
│   │   └── monte_carlo.py
│   ├── greeks.py
│   ├── implied_vol.py
│   ├── hedging.py
│   └── data.py
├── notebooks/
│   └── analysis.ipynb        # generates every plot in this README
├── tests/
│   └── test_pricing.py       # 19 tests: parity, convergence, boundaries, Greeks, IV round-trips
├── streamlit_app.py           # live demo: IV smile + hedging simulator
├── requirements.txt
└── LICENSE
```

Pricing logic lives in tested, importable modules under `src/` — notebooks
are for analysis and plots, not where the math lives. That split is
deliberate: it's what makes the pricers reusable by the Streamlit app,
the test suite, and (if you fork this) anything else, instead of being
locked inside one notebook.

## Running it

```bash
pip install -r requirements.txt

# Run the test suite
pytest tests/ -v

# Regenerate the analysis notebook's outputs and plots
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

# Launch the live demo
streamlit run streamlit_app.py
```

## Limitations

- **European exercise assumption for the Black-Scholes baseline.** American
  early-exercise value is only captured by the binomial tree branch
  (`american=True`), not by the BS or Monte Carlo pricers.
- **No discrete dividends.** Dividend yield `q` is modeled as continuous;
  real equities pay discrete dividends that create small jumps the
  continuous-yield approximation doesn't capture, particularly for
  short-dated options around an ex-dividend date.
- **Constant volatility assumption.** Black-Scholes and the tree both
  assume a single constant σ. The IV smile results (both the notebook's
  synthetic recovery and the live Streamlit tab) are precisely a
  demonstration of where that assumption breaks down against real
  (or realistically-shaped) market prices — the smile itself is evidence
  the constant-vol model is wrong, which is the point of computing it.
- **IV solver has no smoothing or surface fit.** Each strike's IV is
  solved independently via Newton-Raphson/Brent on a raw mid-price; there's
  no SVI or spline fit across strikes. A noisy low-volume strike that
  passes the liquidity filters in `data.py` can still produce a
  "converged" but economically noisy IV — nothing here smooths that out.
- **Live data quality (`data.py`).** Yahoo Finance options data can include
  stale last-trade prices and wide bid-ask spreads on illiquid strikes.
  `get_option_chain` filters on minimum volume, minimum open interest, and
  maximum relative spread, but this is a heuristic filter, not a guarantee
  of quote quality — thin names or far-dated expiries can still return few
  or no strikes after filtering.
- **Hedging simulation ignores transaction costs by default.** The delta
  rebalancing in `hedging.py` supports a `transaction_cost_bps` parameter,
  but the headline result above uses `0` (frictionless). Real hedging P&L
  is worse than shown here once trading costs are included, especially at
  high rebalancing frequencies where the "shrinking error vs. more
  frequent trading" tradeoff has a real cost side that this default view
  doesn't show.
- **Real-world drift vs. risk-neutral drift.** The hedging simulation
  advances the underlying under the risk-neutral measure (drift = r) for
  simplicity. A real hedger's P&L would reflect the underlying's actual
  real-world drift, which need not equal r.

## License

MIT — see [LICENSE](LICENSE).
