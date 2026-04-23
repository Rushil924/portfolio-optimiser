import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Monte Carlo Portfolio Optimiser", layout="wide")
st.title("Monte Carlo Efficient Frontier")

# =====================================================
# Upload standardised returns.xlsx
# =====================================================
uploaded_file = st.file_uploader(
    "Upload standardised price workbook (returns.xlsx)",
    type=["xlsx"]
)

if uploaded_file:

    prices = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()

    st.success("returns.xlsx loaded successfully")

    asset_names = prices.columns.tolist()
    n_assets = len(asset_names)

    # =====================================================
    # Sidebar controls
    # =====================================================
    with st.sidebar:
        st.header("Simulation Settings")
        rf = st.number_input("Risk‑free rate", value=0.0617)
        n_sims = st.slider("Number of simulations", 1_000, 50_000, 10_000)

    # =====================================================
    # Bounds table (TEAM INPUT)
    # =====================================================
    st.subheader("Portfolio Constraints (Editable)")

    bounds_df = pd.DataFrame(
        {
            "Asset": asset_names,
            "Min Weight": [0.0] * n_assets,
            "Max Weight": [1.0 / n_assets * 2] * n_assets
        }
    )

    bounds_df = st.data_editor(bounds_df, num_rows="fixed")

    min_w = bounds_df["Min Weight"].values
    max_w = bounds_df["Max Weight"].values

    # =====================================================
    # Monte‑Carlo engine
    # =====================================================
    def generate_weights(min_w, max_w):
        w = min_w.copy()
        remaining = 1 - w.sum()
        capacity = max_w - min_w

        if remaining < 0 or capacity.sum() < remaining:
            raise ValueError("Infeasible bounds")

        while remaining > 1e-10:
            i = np.random.randint(len(w))
            add = min(capacity[i], np.random.rand() * remaining)
            w[i] += add
            capacity[i] -= add
            remaining -= add

        return w

    def portfolio_stats(returns, w, rf):
        mu = returns.mean() * 12
        cov = returns.cov() * 12

        ret = np.dot(w, mu)
        vol = np.sqrt(w @ cov @ w)
        sharpe = (ret - rf) / vol if vol > 0 else 0

        return ret, vol, sharpe

    # =====================================================
    # Run button
    # =====================================================
    if st.button("Run Monte Carlo Simulation"):

        results = []
        weights = []

        for _ in range(n_sims):
            w = generate_weights(min_w, max_w)
            r, v, s = portfolio_stats(returns, w, rf)
            results.append([r, v, s])
            weights.append(w)

        sim_df = pd.DataFrame(
            results, columns=["Return", "Volatility", "Sharpe"]
        )
        weights_df = pd.DataFrame(weights, columns=asset_names)
        out = pd.concat([sim_df, weights_df], axis=1)

        # =====================================================
        # Max‑Sharpe portfolio
        # =====================================================
        idx = out["Sharpe"].idxmax()
        opt = out.loc[idx]

        # =====================================================
        # Efficient frontier
        # =====================================================
        fig = px.scatter(
            out,
            x="Volatility",
            y="Return",
            color="Sharpe",
            title="Efficient Frontier – Monte Carlo",
            hover_data=asset_names
        )

        fig.add_scatter(
            x=[opt["Volatility"]],
            y=[opt["Return"]],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name="Maximum Sharpe"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # Results
        # =====================================================
        st.subheader("✅ Maximum Sharpe Portfolio")

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Return (pa)", f"{opt['Return']:.2%}")
        col2.metric("Volatility (pa)", f"{opt['Volatility']:.2%}")
        col3.metric("Sharpe Ratio", f"{opt['Sharpe']:.2f}")

        st.subheader("Optimal Weights")

        weights_table = (
            opt[asset_names]
            .to_frame("Weight")
            .style.format("{:.2%}")
        )

        st.dataframe(weights_table)