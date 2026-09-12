# 📈 **Marketing Budget Optimization with MMM**

An **explainable, end-to-end Marketing Mix Modeling (MMM) system** that helps businesses understand **which marketing channels drive incremental sales** and **how to optimize marketing budgets** using data-driven insights.

The project combines **classical MMM techniques** with a **modular Python backend** and an **interactive Streamlit dashboard** designed for real-world decision-making.

---

## 🚀 Business Objective

Marketing leaders need clear answers to:

- Which channels actually drive incremental sales?
- Why does a high-spend channel sometimes show low ROI?
- Where should additional budget be allocated?
- How can these decisions be explained transparently?

This system answers these questions using **interpretable models**, **stable attribution**, and **explainable metrics**.

---

## 🧠 Modeling Approach (High Level)

### Marketing Mix Modeling (MMM)
Sales are modeled as a combination of:
- Baseline demand (trend + seasonality)
- Marketing channel impact
- External controls (price, promotions, holidays)

### Key Techniques
- **Adstock** – captures carryover effects of channels like TV and Video  
- **Saturation** – models diminishing returns at higher spend levels  
- **Ridge Regression** – stabilizes attribution under multicollinearity  
- **Explainable Decomposition** – exposes channel contributions, ROI, and budget guidance  

This ensures the model is **decision-ready**, not just predictive.

---

## 🏗️ Project Architecture

```text
market-mix-modelling/
│
├── app/
│   └── streamlit_app.py        # Interactive Streamlit dashboard
│
├── src/
│   ├── features/
│   │   ├── adstock.py          # Adstock transformations
│   │   ├── saturation.py       # Diminishing returns modeling
│   │   └── seasonality.py      # Trend & seasonal features
│   │
│   ├── models/
│   │   ├── ols_mmm.py          # Baseline OLS MMM
│   │   └── ridge_mmm.py        # Regularized MMM model
│   │
│   ├── analysis/
│   │   ├── contributions.py   # Channel contribution analysis
│   │   ├── roi.py              # ROI & efficiency metrics
│   │   └── scenarios.py        # Budget allocation insights
│   │
│   └── visualization/
│       └── plots.py            # Reusable plotting utilities
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_validation.ipynb
│
├── reports/
│   ├── figures/
│   └── case_study.md
│
├── requirements.txt
└── README.md


---

## 📊 Input Data Schema

The dashboard expects a **time-series CSV dataset** (weekly recommended).

### Required Columns

| Column | Description |
|------|------------|
| `week` | Time period |
| `sales` | Business outcome (revenue / conversions / units) |

### Marketing Channel Spend

| Column | Description |
|------|------------|
| `sp_search` | Search advertising spend |
| `sp_social` | Social media spend |
| `sp_video` | Online video / OTT spend |
| `sp_tv` | Television advertising spend |

### Control Variables (Recommended)

| Column | Description |
|------|------------|
| `promo` | Promotion flag (0/1) |
| `holiday` | Holiday flag (0/1) |
| `price_index` | Relative price index |

Controls prevent incorrect attribution of non-marketing effects.

---

## 🖥️ Streamlit Dashboard Features

- **Overview** – Actual vs Predicted sales  
- **Channel Contributions** – Incremental impact by channel  
- **ROI Analysis** – Average ROI per channel  
- **Budget Guidance** – Identify over- and under-invested channels  

All insights are presented with **plain-English explanations**.

---

## ⚙️ How to Run

### 1️⃣ Setup Environment
```bash
python -m venv .venv

pip install -r requirements.txt

streamlit run app/streamlit_app.py



## 📌 Key Takeaways

- **High contribution ≠ high ROI**  
  A channel can drive large absolute sales while still being inefficient in terms of return per dollar spent.

- **Budget decisions should be driven by marginal ROI**  
  Optimization should focus on where the *next* dollar generates the highest incremental return, not historical averages.

- **Stable attribution matters more than minimal error**  
  Reliable and consistent channel attribution is more valuable for decision-making than aggressively minimizing prediction error.

- **Explainability is critical for stakeholder trust**  
  Clear, transparent reasoning behind model outputs enables confident and defensible business decisions.
