# Machine Learning Guide for Tamil Nadu Agricultural Project

## Quick Reference

| Concept | Meaning |
|---------|---------|
| Supervised Learning | Learning from labeled examples |
| Regression | Predicting numbers (like price) |
| Grid Search | Test ALL hyperparameter combinations |
| Randomized Search | Test RANDOM combinations (faster) |
| Overfitting | Model memorizes, doesn't generalize |
| Underfitting | Model too simple to learn |
| Feature Importance | Which inputs matter most |

---

## Your Model Summary

| Metric | Value |
|--------|-------|
| Model Type | Gradient Boosting |
| Accuracy (R²) | 95.85% |
| Uses Lag Features | NO ✅ |
| Uses Live Weather | YES ✅ |

---

## Feature Importance

```
commodity_encoded   ████████████████████████ 92.79%
year_trend          ██ 6.47%
rainfall_mm         █ 0.32%
month               █ 0.27%
```

**Key Insight:** Commodity type is the strongest predictor (93%)

---

## How to Run Predictions

```bash
# Live predictions with weather
python scripts/live_prediction.py

# Train model
python scripts/tn_no_lag_model.py
```

---

## Project Files

| File | Purpose |
|------|---------|
| `live_prediction.py` | Real-time predictions with weather API |
| `tn_no_lag_model.py` | Main ML model (no lag features) |
| `tn_ml_pipeline.py` | Complete ML pipeline |
| `data_quality.py` | Data preprocessing |
| `eda_analysis.py` | Exploratory analysis |
