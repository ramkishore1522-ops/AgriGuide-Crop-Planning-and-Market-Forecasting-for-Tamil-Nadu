"""
PROPHET MODELING (Per-Commodity)
==================================
Uses Facebook Prophet to model time series with climate features as additional regressors.
Evaluated using the same expanding-window cross-validation for a fair comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import sys
import io
import warnings

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("Prophet is not installed. Install with `pip install prophet`.")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
REPORTS = PROJECT_ROOT / 'reports'

# IEEE Plot Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300})

def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def load_data() -> pd.DataFrame:
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    prices['date'] = pd.to_datetime(prices['date'])
    tn = prices[prices['state_name'] == 'Tamil Nadu'].copy()
    tn['year'] = tn['date'].dt.year
    tn['month'] = tn['date'].dt.month
    price_monthly = tn.groupby(['commodity', 'year', 'month']).agg(price=('price', 'mean')).reset_index()

    try:
        rain_raw = pd.read_csv(DATA_RAW / 'daily-rainfall-data-district-level.csv')
        tn_rain = rain_raw[rain_raw['state_name'].str.contains('Tamil', case=False, na=False)].copy()
        tn_rain['date'] = pd.to_datetime(tn_rain['date'])
    except:
        rain_raw = pd.read_csv(DATA_QUALITY / 'rainfall_state_quality.csv')
        tn_rain = rain_raw[rain_raw['state_name'] == 'Tamil Nadu'].copy()
        tn_rain['date'] = pd.to_datetime(tn_rain['date'])

    tn_rain['year'] = tn_rain['date'].dt.year
    tn_rain['month'] = tn_rain['date'].dt.month

    if 'actual' in tn_rain.columns:
        rain_monthly = tn_rain.groupby(['year', 'month']).agg(
            rainfall_mm=('actual', 'sum'), rainfall_deviation=('deviation', 'mean')
        ).reset_index()
    else:
        rain_monthly = tn_rain.groupby(['year', 'month']).agg(rainfall_mm=('rainfall', 'sum')).reset_index()
        rain_monthly['rainfall_deviation'] = 0.0

    merged = price_monthly.merge(rain_monthly, on=['year', 'month'], how='left')
    merged['rainfall_mm'] = merged['rainfall_mm'].fillna(merged['rainfall_mm'].median())
    merged['rainfall_deviation'] = merged['rainfall_deviation'].fillna(0)
    merged['date'] = pd.to_datetime(merged['year'].astype(str) + '-' + merged['month'].astype(str) + '-01')
    return merged.sort_values(['commodity', 'date']).reset_index(drop=True)

def run_prophet_cv(df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_PROPHET:
        return pd.DataFrame()
        
    print("\n" + "=" * 70)
    print("PROPHET MODELING (Expanding Window CV)")
    print("=" * 70)
    
    years = sorted(df['year'].unique())
    min_train_years = 4
    
    results = []
    
    for commodity in df['commodity'].unique():
        comm_df = df[df['commodity'] == commodity].copy()
        
        # Prophet requires 'ds' and 'y'
        comm_df['ds'] = comm_df['date']
        comm_df['y'] = comm_df['price']
        
        print(f"\n  ── {commodity} ──")
        
        for fold_idx in range(min_train_years, len(years)):
            train_years = years[:fold_idx]
            test_year = years[fold_idx]
            
            train = comm_df[comm_df['year'].isin(train_years)].copy()
            test = comm_df[comm_df['year'] == test_year].copy()
            
            if len(train) < 24 or len(test) < 3:
                continue
                
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            # Add climate regressors
            model.add_regressor('rainfall_mm')
            model.add_regressor('rainfall_deviation')
            
            try:
                model.fit(train[['ds', 'y', 'rainfall_mm', 'rainfall_deviation']])
                
                # Predict
                future = test[['ds', 'rainfall_mm', 'rainfall_deviation']].copy()
                forecast = model.predict(future)
                
                preds = forecast['yhat'].values
                actuals = test['y'].values
                
                results.append({
                    'Commodity': commodity,
                    'Fold': f"{train_years[-1]}→{test_year}",
                    'R2': r2_score(actuals, preds),
                    'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
                    'MAE': mean_absolute_error(actuals, preds),
                    'MAPE': mape(actuals, preds)
                })
            except Exception as e:
                print(f"    Failed fold {train_years[-1]}→{test_year}: {e}")
                
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        summary = res_df.groupby('Commodity').agg({
            'R2': ['mean', 'std'], 'RMSE': 'mean', 'MAE': 'mean', 'MAPE': 'mean'
        }).round(3)
        summary.columns = ['R2_mean', 'R2_std', 'RMSE_mean', 'MAE_mean', 'MAPE_mean']
        print("\nProphet Results Summary:")
        print(summary.to_string())
        
        res_df.to_csv(REPORTS / 'prophet_results.csv', index=False)
        print(f"\nSaved to: {REPORTS / 'prophet_results.csv'}")
        
    return res_df

if __name__ == '__main__':
    df = load_data()
    run_prophet_cv(df)
