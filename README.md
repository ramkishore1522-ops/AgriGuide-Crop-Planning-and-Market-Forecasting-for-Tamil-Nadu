# 🌾 AgriGuide: Tamil Nadu Agricultural Price Prediction

![Dashboard Preview](https://img.shields.io/badge/Status-Deployment%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Model-Gradient%20Boosting-orange)

[**➡️ VIEW LIVE DASHBOARD DEMO HERE ⬅️**]([https://agriguide-crop-planning-and-market-forecasting-for-tamil-nadu.streamlit.app/]))

AgriGuide is an advanced machine learning framework designed to predict retail prices for essential agricultural commodities across the 32 districts of Tamil Nadu, India. 

Developed specifically for academic and open-source publication, this project introduces a **Gradient Boosting Model** capable of robust forecasting without relying on historical price lags, and leverages **Conformal Prediction** to mathematically guarantee uncertainty boundaries on future prices.

---

## 🌟 Key Features
- **Gradient Boosting Architecture:** Robust predictive performance (R² > 95%) that functions dynamically without needing previous months' prices, allowing for long-term forecasting.
- **Conformal Prediction:** Provides 90% and 95% confidence intervals, giving policymakers mathematically sound boundaries for agricultural inflation.
- **Live Weather Integration:** Real-time rainfall and temperature data integration via the Open-Meteo API.
- **Granger Causality Testing:** Rigorous statistical proofs showing the exact lag at which rainfall deviations trigger price hikes.

---

## 🚀 Live Dashboard Deployment

This repository is pre-configured to be deployed immediately to **Streamlit Community Cloud**.

1. Fork or push this repository to your own GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select this repository and point the main file path to `dashboard.py`.
4. Click **Deploy!**

The dashboard utilizes a lightweight `4.5 MB` dataset (`tn_retail_prices_dashboard.csv`) extracted from the main 125 MB raw data to ensure instant cloud loading speeds without crashing server memory.

---

## 💻 Reproducing Publication Results (Local Setup)

To completely reproduce all models, tables, and statistical analyses from the associated publication, run the master reproduction script.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
```bash
python reproduce_all.py
```
This master orchestrator will sequentially execute:
1. `tn_no_lag_model.py` (Core Gradient Boosting model)
2. `error_analysis.py` (Residual & error breakdown)
3. `per_commodity_pipeline.py` (Baseline vs GB models)
4. `conformal_prediction.py` (Uncertainty bounds)
5. `granger_causality.py` (Statistical proofs)

All generated artifacts, LaTeX tables, and plots will automatically be saved to the `reports/` and `visualizations/` directories.

---

## 📁 Repository Structure
```text
AgriGuide/
├── scripts/
│   ├── 01_data_processing/    ← Cleaning & imputation
│   ├── 02_modeling/           ← Hybrid model training
│   ├── 03_evaluation/         ← Conformal & Granger tests
│   └── 04_inference/          ← Terminal predictors
├── models/
│   └── tn_no_lag_model.joblib ← Compressed trained weights
├── data/
│   └── quality_checked/       ← (Ignored via .gitignore except dashboard data)
├── reports/                   ← Final LaTeX tables & CSV results
├── visualizations/            ← High-res figures for publication
├── dashboard.py               ← Streamlit Web App
└── reproduce_all.py           ← Master Pipeline Orchestrator
```

## ⚖️ License & Data
The historical price data was originally sourced from public government portals (Agmarknet). Please refer to `LICENSE` for distribution rights.
