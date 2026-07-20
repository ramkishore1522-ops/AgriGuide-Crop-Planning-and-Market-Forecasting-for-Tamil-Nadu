"""
LSTM MODELING (Per-Commodity)
===============================
Uses PyTorch to model time series with climate features via an LSTM network.
Evaluated using the same expanding-window cross-validation for a fair comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch is not installed. Install with `pip install torch`.")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
REPORTS = PROJECT_ROOT / 'reports'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)

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
    
    # Feature engineering
    merged['year_trend'] = merged['year'] - merged['year'].min()
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
    merged['season'] = merged['month'].apply(
        lambda m: 0 if m in [6, 7, 8, 9] else (1 if m in [10, 11] else (2 if m in [12, 1, 2] else 3))
    )
    
    merged['date'] = pd.to_datetime(merged['year'].astype(str) + '-' + merged['month'].astype(str) + '-01')
    return merged.sort_values(['commodity', 'date']).reset_index(drop=True)

if HAS_TORCH:
    class PriceLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

def create_sequences(data, target, lookback=6):
    Xs, ys = [], []
    for i in range(len(data) - lookback):
        Xs.append(data[i:(i + lookback)])
        ys.append(target[i + lookback])
    return np.array(Xs), np.array(ys)

def run_lstm_cv(df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_TORCH:
        return pd.DataFrame()
        
    print("\n" + "=" * 70)
    print("LSTM MODELING (Expanding Window CV)")
    print("=" * 70)
    
    features = ['price', 'rainfall_mm', 'rainfall_deviation', 'year_trend', 'month_sin', 'month_cos', 'season']
    lookback = 6
    years = sorted(df['year'].unique())
    min_train_years = 4
    
    results = []
    
    for commodity in df['commodity'].unique():
        comm_df = df[df['commodity'] == commodity].copy()
        print(f"\n  ── {commodity} ──")
        
        for fold_idx in range(min_train_years, len(years)):
            train_years = years[:fold_idx]
            test_year = years[fold_idx]
            
            # Need to include the end of train for the lookback window of test
            train = comm_df[comm_df['year'].isin(train_years)].copy()
            test_raw = comm_df[comm_df['year'] == test_year].copy()
            
            if len(train) < lookback + 12 or len(test_raw) < 1:
                continue
                
            test = pd.concat([train.tail(lookback), test_raw])
            
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            train_X_scaled = scaler_X.fit_transform(train[features])
            train_y_scaled = scaler_y.fit_transform(train[['price']])
            test_X_scaled = scaler_X.transform(test[features])
            test_y_scaled = scaler_y.transform(test[['price']])
            
            X_train_seq, y_train_seq = create_sequences(train_X_scaled, train_y_scaled, lookback)
            X_test_seq, y_test_seq = create_sequences(test_X_scaled, test_y_scaled, lookback)
            
            if len(X_test_seq) == 0:
                continue
                
            train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            model = PriceLSTM(input_size=len(features))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Train
            model.train()
            for epoch in range(100):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())
                    loss.backward()
                    optimizer.step()
                    
            # Eval
            model.eval()
            with torch.no_grad():
                preds_scaled = model(torch.FloatTensor(X_test_seq)).numpy()
                preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                actuals = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
                
            results.append({
                'Commodity': commodity,
                'Fold': f"{train_years[-1]}→{test_year}",
                'R2': r2_score(actuals, preds),
                'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
                'MAE': mean_absolute_error(actuals, preds),
                'MAPE': mape(actuals, preds)
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        summary = res_df.groupby('Commodity').agg({
            'R2': ['mean', 'std'], 'RMSE': 'mean', 'MAE': 'mean', 'MAPE': 'mean'
        }).round(3)
        summary.columns = ['R2_mean', 'R2_std', 'RMSE_mean', 'MAE_mean', 'MAPE_mean']
        print("\nLSTM Results Summary:")
        print(summary.to_string())
        
        res_df.to_csv(REPORTS / 'lstm_results.csv', index=False)
        print(f"\nSaved to: {REPORTS / 'lstm_results.csv'}")
        
    return res_df

if __name__ == '__main__':
    df = load_data()
    run_lstm_cv(df)
