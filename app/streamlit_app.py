import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import streamlit as st
import pandas as pd
import numpy as np

# Internal imports
from src.features.adstock import geometric_adstock
from src.features.saturation import hill_saturation
from src.features.seasonality import add_fourier_seasonality

from src.models.ridge_model import fit_ridge_mmm
from src.analysis.contribution import compute_contributions
from src.analysis.roi import compute_roi

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Explainable MMM Dashboard", layout="wide")

st.title("📊 Explainable Marketing Mix Modeling Dashboard")
st.markdown("""
This dashboard helps you understand **which marketing channels drive sales**  
and **where to invest your next marketing dollar**.
""")

# -----------------------------
# Upload data
# -----------------------------
st.sidebar.header("📂 Upload MMM Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your MMM dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file, parse_dates=["week"])
df = df.sort_values("week").reset_index(drop=True)

# -----------------------------
# Feature selection
# -----------------------------
st.sidebar.header("⚙️ Configuration")

target_col = st.sidebar.selectbox("Select target (sales)", ["sales"])
channel_cols = ["sp_search", "sp_social", "sp_video", "sp_tv"]

# -----------------------------
# Baseline + seasonality
# -----------------------------
df["t"] = np.arange(len(df))
df = add_fourier_seasonality(df, period=52, order=2)

# -----------------------------
# Adstock + Saturation
# -----------------------------
adstock_params = {
    "sp_search": 0.3,
    "sp_social": 0.5,
    "sp_video": 0.7,
    "sp_tv": 0.85,
}

for ch, decay in adstock_params.items():
    df[f"{ch}_ad"] = geometric_adstock(df[ch].values, decay)
    df[f"{ch}_sat"] = hill_saturation(
        df[f"{ch}_ad"].values,
        alpha=1.3,
        ec50=df[f"{ch}_ad"].mean()
    )

# -----------------------------
# Model matrix
# -----------------------------
feature_cols = (
    [f"{ch}_sat" for ch in channel_cols]
    + ["promo", "holiday", "price_index"]
    + [c for c in df.columns if c.startswith("sin_") or c.startswith("cos_")]
    + ["t"]
)

X = df[feature_cols]
y = df[target_col]

# -----------------------------
# Fit Ridge MMM
# -----------------------------
model = fit_ridge_mmm(X, y, alpha=1.0)
df["prediction"] = model.predict(X)

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview", "🧩 Contributions", "💰 ROI", "🎯 Recommendations"]
)

# =============================
# TAB 1 — Overview
# =============================
with tab1:
    st.subheader("Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df["week"], df["sales"], label="Actual")
    ax.plot(df["week"], df["prediction"], label="Predicted")
    ax.legend()
    ax.set_title("Model Fit")
    st.pyplot(fig)

    st.markdown("""
    **What this shows:**  
    The model captures overall trends, seasonality, and marketing-driven lifts.
    """)

# =============================
# TAB 2 — Contributions
# =============================
with tab2:
    st.subheader("Channel Contribution Decomposition")

    contrib_df = compute_contributions(
        model=model,
        X=X,
        channel_cols=[f"{ch}_sat" for ch in channel_cols]
    )

    channel_contrib = contrib_df[[f"{ch}_sat" for ch in channel_cols]].sum()

    fig, ax = plt.subplots(figsize=(10,4))
    channel_contrib.plot(kind="bar", ax=ax)
    ax.set_title("Total Incremental Contribution by Channel")
    ax.set_ylabel("Incremental Sales")
    st.pyplot(fig)

    st.markdown("""
    **Explanation:**  
    This shows how much incremental sales each channel contributed historically.
    """)

# =============================
# TAB 3 — ROI
# =============================
with tab3:
    st.subheader("Return on Investment (ROI)")

    roi_df = compute_roi(
        df=df,
        contrib_df=contrib_df,
        spend_cols=channel_cols
    )

    st.dataframe(roi_df)

    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(data=roi_df, x="channel", y="roi", ax=ax)
    ax.set_title("Average ROI by Channel")
    ax.set_ylabel("Incremental Sales per $ Spent")
    st.pyplot(fig)

    st.markdown("""
    **How to read this:**  
    - High ROI → efficient channel  
    - Low ROI → expensive scale channel  
    """)

# =============================
# TAB 4 — Recommendations
# =============================
with tab4:
    st.subheader("Budget Optimization Guidance")

    st.markdown("""
    **Key Insight:**  
    Budget decisions should be driven by **marginal ROI**, not total revenue.
    """)

    best_channel = roi_df.iloc[0]["channel"]
    worst_channel = roi_df.iloc[-1]["channel"]

    st.success(f"✅ Increase investment in **{best_channel.upper()}**")
    st.warning(f"⚠️ Consider reducing investment in **{worst_channel.upper()}**")

    st.markdown("""
    **Why this works:**  
    - Some channels scale well but saturate early (TV)  
    - Others remain efficient at higher spend (Video, Social)  
    - MMM identifies *where the next dollar works hardest*
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("📌 *Explainable AI MMM Dashboard — Built with Ridge Regression, Adstock & Saturation*")
