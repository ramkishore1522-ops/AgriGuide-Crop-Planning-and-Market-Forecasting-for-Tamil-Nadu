"""
PUBLICATION ML PIPELINE
Advanced machine learning pipeline with statistical rigor, cross-validation,
and explainability (SHAP) required for IEEE/Scopus publications.
Now includes LSTM for sequence modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import io
import warnings
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ML and Stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Plot styling for IEEE (300 DPI, Serif)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
MODELS_DIR = PROJECT_ROOT / 'models'
VISUALIZATIONS = PROJECT_ROOT / 'visualizations'
REPORTS = PROJECT_ROOT / 'reports'

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE avoiding division by zero"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_and_prepare_data():
    """Load and prepare data for ML models"""
    print("Loading data...")
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    prices['date'] = pd.to_datetime(prices['date'])
    
    # We use a state-level aggregation for generalized modelling
    df = prices.groupby(['date', 'commodity']).agg({'price': 'mean'}).reset_index()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Load rainfall and merge (state-level approximation)
    rainfall = pd.read_csv(DATA_QUALITY / 'rainfall_state_quality.csv')
    rainfall['date'] = pd.to_datetime(rainfall['date'])
    rainfall['year'] = rainfall['date'].dt.year
    rainfall['month'] = rainfall['date'].dt.month
    
    rain_agg = rainfall.groupby(['year', 'month']).agg({
        'actual': 'mean',
        'deviation': 'mean'
    }).reset_index()
    
    merged = df.merge(rain_agg, on=['year', 'month'], how='left')
    merged['actual'] = merged['actual'].fillna(merged['actual'].median())
    merged['deviation'] = merged['deviation'].fillna(0)
    
    # Feature Engineering
    le = LabelEncoder()
    merged['commodity_idx'] = le.fit_transform(merged['commodity'])
    merged['season'] = merged['month'].apply(lambda x: 0 if x in [6,7,8,9] else (1 if x in [10,11] else (2 if x in [12,1,2] else 3)))
    merged['is_monsoon'] = (merged['month'].isin([6, 7, 8, 9])).astype(int)
    
    features = ['commodity_idx', 'year', 'month', 'season', 'is_monsoon', 'actual', 'deviation']
    
    # Sort chronologically globally
    merged = merged.sort_values('date').reset_index(drop=True)
    
    return merged, features, le


def create_sequences_chronological(df, features, target='price', time_steps=3):
    """Create 3D sequences globally sorted by date to preserve TimeSeriesSplit logic"""
    Xs, ys, dates = [], [], []
    
    # Group by commodity to form sequences, but keep track of target date
    for commodity in df['commodity'].unique():
        comm_data = df[df['commodity'] == commodity].sort_values('date')
        
        if len(comm_data) <= time_steps:
            continue
            
        x_vals = comm_data[features].values
        y_vals = comm_data[target].values
        d_vals = comm_data['date'].values
        
        for i in range(len(comm_data) - time_steps):
            Xs.append(x_vals[i:(i + time_steps)])
            ys.append(y_vals[i + time_steps])
            dates.append(d_vals[i + time_steps])
            
    # Convert to arrays
    Xs = np.array(Xs)
    ys = np.array(ys)
    dates = np.array(dates)
    
    # Sort everything globally by date
    sort_idx = np.argsort(dates)
    
    return Xs[sort_idx], ys[sort_idx]


def build_lstm_model(timesteps, features):
    """Build and compile LSTM architecture"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def run_timeseries_cv(df, features, time_steps=3):
    """Run rigorous TimeSeries Cross-Validation and statistical tests"""
    print("\nRunning TimeSeries Cross-Validation...")
    
    # Generate globally chronologically sorted sequential data
    X_seq, y = create_sequences_chronological(df, features, time_steps=time_steps)
    
    # For traditional ML, we flatten to the most recent timestep in the sequence
    X_flat = X_seq[:, -1, :] 
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'Naive Baseline': None,
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
        'LSTM (Keras)': 'lstm_placeholder'
    }
    
    results = {name: {'r2': [], 'rmse': [], 'mape': [], 'preds': []} for name in models.keys()}
    scaler = StandardScaler()
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_flat)):
        
        # Traditional ML splits
        X_train_flat, X_test_flat = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_sc = scaler.fit_transform(X_train_flat)
        X_test_sc = scaler.transform(X_test_flat)
        
        # LSTM scaling - simple 2D scaling applied to 3D via reshape
        X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
        
        samples_tr, ts_tr, feats_tr = X_train_seq.shape
        samples_te, ts_te, feats_te = X_test_seq.shape
        
        # Reshape to 2D for scaler
        X_train_seq_2d = X_train_seq.reshape(-1, feats_tr)
        X_test_seq_2d = X_test_seq.reshape(-1, feats_te)
        
        seq_scaler = StandardScaler()
        X_train_seq_2d_sc = seq_scaler.fit_transform(X_train_seq_2d)
        X_test_seq_2d_sc = seq_scaler.transform(X_test_seq_2d)
        
        # Reshape back to 3D
        X_train_seq_sc = X_train_seq_2d_sc.reshape(samples_tr, ts_tr, feats_tr)
        X_test_seq_sc = X_test_seq_2d_sc.reshape(samples_te, ts_te, feats_te)
        
        for name, model in models.items():
            if name == 'Naive Baseline':
                preds = np.full(len(y_test), y_train.mean())
            elif name == 'LSTM (Keras)':
                lstm_model = build_lstm_model(time_steps, len(features))
                es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                lstm_model.fit(X_train_seq_sc, y_train, epochs=50, batch_size=32, 
                               validation_split=0.2, callbacks=[es], verbose=0)
                preds = lstm_model.predict(X_test_seq_sc, verbose=0).flatten()
            else:
                model.fit(X_train_sc, y_train)
                preds = model.predict(X_test_sc)
            
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mape = mean_absolute_percentage_error(y_test, preds)
            
            results[name]['r2'].append(r2)
            results[name]['rmse'].append(rmse)
            results[name]['mape'].append(mape)
            results[name]['preds'].extend(preds)
            
        print(f"Completed Fold {fold+1}/5")
    
    # Statistical significance
    best_model_name = max(results.keys(), key=lambda k: np.mean(results[k]['r2']))
    best_preds = results[best_model_name]['preds']
    
    summary = []
    print("\nModel Performance (Mean ± Std across 5 folds):")
    for name in models.keys():
        r2_mean, r2_std = np.mean(results[name]['r2']), np.std(results[name]['r2'])
        rmse_mean, rmse_std = np.mean(results[name]['rmse']), np.std(results[name]['rmse'])
        mape_mean, mape_std = np.mean(results[name]['mape']), np.std(results[name]['mape'])
        
        if name != best_model_name:
            _, p_val = stats.wilcoxon(results[name]['preds'], best_preds)
            sig = f" (p={p_val:.3e})"
        else:
            sig = " (Best Model)"
            
        summary.append({
            'Model': name,
            'R2': f"{r2_mean:.3f} ± {r2_std:.3f}",
            'RMSE': f"{rmse_mean:.2f} ± {rmse_std:.2f}",
            'MAPE': f"{mape_mean:.2f}% ± {mape_std:.2f}%",
            'Significance': sig.strip()
        })
        print(f"{name:15s}: R2={r2_mean:.3f}, RMSE={rmse_mean:.2f}, MAPE={mape_mean:.2f}% {sig}")
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(REPORTS / 'table_model_comparison.csv', index=False)
    
    # Fit final model
    X_sc = scaler.fit_transform(X_flat)
    
    best_tree_name = 'XGBoost' if best_model_name in ['LSTM (Keras)', 'Naive Baseline'] else best_model_name
    final_model = models[best_tree_name]
    
    if final_model is not None:
        final_model.fit(X_sc, y)
    else:
        # Fallback if best tree was naive (which shouldn't happen)
        final_model = models['XGBoost']
        final_model.fit(X_sc, y)
        
    return final_model, scaler, X_sc, features


def generate_shap_analysis(model, X_scaled, features):
    """Generate SHAP values for model explainability"""
    print("\nGenerating SHAP explainability analysis...")
    try:
        background = shap.sample(X_scaled, 100)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X_scaled[:500]) 
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features=features, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS / 'figure_4_shap_summary.png', dpi=300)
        plt.close()
        print(f"Saved SHAP summary to {VISUALIZATIONS / 'figure_4_shap_summary.png'}")
        
    except Exception as e:
        print(f"SHAP generation failed: {e}")


def main():
    print("="*70)
    print("PUBLICATION ML PIPELINE (IEEE/SCOPUS STANDARD)")
    print("="*70)
    
    df, features, le = load_and_prepare_data()
    best_model, scaler, X_sc, feature_names = run_timeseries_cv(df, features)
    generate_shap_analysis(best_model, X_sc, feature_names)
    
    print("\nPipeline Complete. Models and figures saved.")

if __name__ == '__main__':
    main()
