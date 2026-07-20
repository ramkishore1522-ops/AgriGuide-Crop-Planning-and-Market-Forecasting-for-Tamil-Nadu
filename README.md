# Tamil Nadu Agricultural Price Prediction System

## Setup Instructions (Step by Step)

### Step 1: Install Python
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, **CHECK "Add Python to PATH"** ← Very important!
3. Click Install

### Step 2: Extract the Project
1. Extract the ZIP file to any folder (e.g., Desktop)
2. Open the folder → you should see: `scripts/`, `models/`, `data/`, etc.

### Step 3: Open Terminal
1. Open the project folder
2. Click on the address bar, type `cmd`, press Enter
   - OR open Command Prompt and type: `cd path\to\Project-2`

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
Wait for all packages to install (2-3 minutes).

### Step 5: Run Predictions
```bash
# Interactive prediction (enter your own inputs)
python scripts/predict.py

# Live weather-based prediction
python scripts/live_prediction.py
```

---

## How to Use

### Interactive Predictor
```
python scripts/predict.py
```
1. Select commodity number (1-22)
2. Enter month (1-12)
3. Enter year (e.g., 2026)
4. Get predicted price!

### Live Weather Prediction
```
python scripts/live_prediction.py
```
- Automatically fetches live weather
- Predicts prices for 7 commodities
- Shows weather data for Tamil Nadu

---

## Reproducing Paper Results

To completely reproduce all models, tables, and analyses from the associated publication, run the master reproduction script. This will sequentially execute the entire machine learning pipeline, evaluate errors, and compute conformal predictions.

```bash
python reproduce_all.py
```

The generated artifacts and figures will be saved in the `reports/` and `visualizations/` directories.

---

## Data Availability

The original datasets used in this project should be placed in the `data/raw/` directory. If the dataset is too large to include in this repository, please download it from the original source (e.g., Agmarknet) and place the raw CSV files in `data/raw/`. The data cleaning scripts will automatically process them into the `data/quality_checked/` folder.

---

## Project Structure
```
Project-2/
├── scripts/
│   ├── predict.py            ← Interactive predictor
│   ├── live_prediction.py    ← Weather API prediction
│   ├── tn_no_lag_model.py    ← ML training code
│   ├── tn_ml_pipeline.py     ← Full pipeline
│   ├── data_quality.py       ← Data cleaning
│   └── eda_analysis.py       ← EDA analysis
├── models/
│   └── tn_no_lag_model.joblib ← Trained model
├── data/
│   ├── raw/                   ← Original datasets
│   ├── quality_checked/       ← Cleaned data
│   └── tamil_nadu/            ← TN filtered data
├── reports/                   ← Analysis reports
├── visualizations/            ← Charts and graphs
├── docs/                      ← Documentation
├── requirements.txt           ← Python packages
└── PROJECT_REVIEW_DOCUMENTATION.md ← Full project report
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python not found` | Reinstall Python, check "Add to PATH" |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` | Make sure you're in the Project-2 folder |
| `No module named sklearn` | Run `pip install scikit-learn` |
