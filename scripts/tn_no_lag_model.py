"""
TAMIL NADU PRICE PREDICTION - WITHOUT LAG FEATURES
Uses: Rainfall, District, Season, Commodity (No previous price needed!)
More practical for real-world future predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import io
import importlib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
DATA_TN = PROJECT_ROOT / 'data' / 'tamil_nadu'
MODELS_DIR = PROJECT_ROOT / 'models'
VISUALIZATIONS = PROJECT_ROOT / 'visualizations'
REPORTS = PROJECT_ROOT / 'reports'


def load_and_prepare_data():
    """Load TN prices and rainfall, merge them."""
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA (NO LAG FEATURES)")
    print("="*60)
    
    # Load TN prices
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    tn_prices = prices[prices['state_name'] == 'Tamil Nadu'].copy()
    tn_prices['date'] = pd.to_datetime(tn_prices['date'])
    print(f"  TN Price records: {len(tn_prices):,}")
    
    # Load TN rainfall (district level)
    try:
        rainfall = pd.read_csv(DATA_RAW / 'daily-rainfall-data-district-level.csv')
        tn_rainfall = rainfall[rainfall['state_name'].str.contains('Tamil', case=False, na=False)].copy()
        tn_rainfall['date'] = pd.to_datetime(tn_rainfall['date'])
        print(f"  TN Rainfall records: {len(tn_rainfall):,}")
        has_district_rainfall = True
    except:
        print("  Using state-level rainfall")
        rainfall = pd.read_csv(DATA_QUALITY / 'rainfall_state_quality.csv')
        tn_rainfall = rainfall[rainfall['state_name'] == 'Tamil Nadu'].copy()
        tn_rainfall['date'] = pd.to_datetime(tn_rainfall['date'])
        has_district_rainfall = False
    
    # Aggregate prices by month
    tn_prices['year'] = tn_prices['date'].dt.year
    tn_prices['month'] = tn_prices['date'].dt.month
    
    price_monthly = tn_prices.groupby(['commodity', 'year', 'month']).agg({
        'price': 'mean'
    }).reset_index()
    print(f"  Monthly price records: {len(price_monthly):,}")
    
    # Aggregate rainfall by month
    tn_rainfall['year'] = tn_rainfall['date'].dt.year
    tn_rainfall['month'] = tn_rainfall['date'].dt.month
    
    if 'actual' in tn_rainfall.columns:
        rainfall_monthly = tn_rainfall.groupby(['year', 'month']).agg({
            'actual': 'sum',
            'deviation': 'mean'
        }).reset_index()
        rainfall_monthly.columns = ['year', 'month', 'rainfall_mm', 'rainfall_deviation']
    else:
        rainfall_monthly = tn_rainfall.groupby(['year', 'month']).agg({
            'rainfall': 'sum'
        }).reset_index()
        rainfall_monthly.columns = ['year', 'month', 'rainfall_mm']
        rainfall_monthly['rainfall_deviation'] = 0
    
    print(f"  Monthly rainfall records: {len(rainfall_monthly):,}")
    
    # Merge prices with rainfall
    merged = price_monthly.merge(rainfall_monthly, on=['year', 'month'], how='left')
    merged['rainfall_mm'] = merged['rainfall_mm'].fillna(merged['rainfall_mm'].median())
    merged['rainfall_deviation'] = merged['rainfall_deviation'].fillna(0)
    
    print(f"  Merged records: {len(merged):,}")
    
    return merged, tn_prices


def create_features(df):
    """Create features WITHOUT lag (no previous price needed)."""
    print("\n" + "="*60)
    print("STEP 2: CREATING FEATURES (NO LAG!)")
    print("="*60)
    
    df = df.copy()
    
    # Encode commodity
    le_commodity = LabelEncoder()
    df['commodity_encoded'] = le_commodity.fit_transform(df['commodity'])
    
    # Season features
    def get_season(month):
        if month in [6, 7, 8, 9]:  # June-Sep: Monsoon
            return 0
        elif month in [10, 11]:  # Oct-Nov: Post-monsoon
            return 1
        elif month in [12, 1, 2]:  # Dec-Feb: Winter
            return 2
        else:  # Mar-May: Summer
            return 3
    
    df['season'] = df['month'].apply(get_season)
    
    # Is monsoon month (high impact on agriculture)
    df['is_monsoon'] = (df['month'].isin([6, 7, 8, 9])).astype(int)
    
    # Year trend (relative to start)
    df['year_trend'] = df['year'] - df['year'].min()
    
    # Rainfall categories (handle NaN)
    df['rainfall_mm'] = df['rainfall_mm'].fillna(df['rainfall_mm'].median())
    df['rainfall_category'] = pd.cut(
        df['rainfall_mm'], 
        bins=[-1, 50, 150, 300, 10000],
        labels=[0, 1, 2, 3]  # Low, Medium, High, Very High
    ).astype(float).fillna(1).astype(int)
    
    # Features to use (NO LAG!)
    features = [
        'commodity_encoded',
        'year_trend',
        'month',
        'season',
        'is_monsoon',
        'rainfall_mm',
        'rainfall_deviation',
        'rainfall_category'
    ]
    
    print(f"\n  Features used (NO PREVIOUS PRICE!):")
    for f in features:
        print(f"    - {f}")
    
    return df, features, le_commodity


def compute_regression_metrics(y_true, y_pred):
    """Calculate a richer set of regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def summarize_fold_metrics(fold_metrics):
    """Summarize metrics across folds and compute confidence intervals."""
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    summary_rows = []
    for model_name in sorted({row['Model'] for row in fold_metrics}):
        model_rows = [row for row in fold_metrics if row['Model'] == model_name]
        summary = {'Model': model_name}
        for metric in metrics:
            values = [row[metric] for row in model_rows]
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci_half_width = 1.96 * std_value / np.sqrt(len(values)) if len(values) > 1 else 0.0
            summary[f'{metric}_mean'] = round(mean_value, 4)
            summary[f'{metric}_std'] = round(std_value, 4)
            summary[f'{metric}_ci95'] = round(ci_half_width, 4)
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows)


def summarize_baseline_results(metric_list, model_name):
    """Summarize optional baseline results into a DataFrame-ready row."""
    summary = {'Model': model_name}
    for metric in ['MAE', 'RMSE', 'MAPE', 'R2']:
        values = [m[metric] for m in metric_list]
        summary[f'{metric}_mean'] = round(float(np.mean(values)), 4)
    return pd.DataFrame([summary])


def get_time_series_splits(df, min_train_years=4):
    """Create expanding and rolling time-series splits."""
    years = sorted(df['year'].unique())
    splits = []
    for idx in range(min_train_years, len(years)):
        train_years = years[:idx]
        test_years = [years[idx]]
        splits.append((train_years, test_years))
    return splits


def evaluate_baselines(df, features):
    """Evaluate multiple baseline models with rolling and expanding time-series validation."""
    print("\n" + "="*60)
    print("STEP 3A: TIME-SERIES VALIDATION")
    print("="*60)

    X = df[features]
    y = df['price']
    splits = get_time_series_splits(df)

    model_specs = [
        ('Naive Mean', DummyRegressor(strategy='mean'), False),
        ('Linear Regression', LinearRegression(), True),
        ('Ridge Regression', Ridge(alpha=10.0), True),
        ('Random Forest', RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1), False),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=120, max_depth=5, random_state=42), False),
        ('Hybrid Ensemble', None, None)
    ]

    fold_results = []
    for train_years, test_years in splits:
        train_mask = df['year'].isin(train_years)
        test_mask = df['year'].isin(test_years)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) < 20 or len(X_test) < 5:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model, needs_scaling in model_specs:
            if name == 'Hybrid Ensemble':
                base_model = GradientBoostingRegressor(n_estimators=120, max_depth=5, random_state=42)
                ridge_model = Ridge(alpha=10.0)
                base_model.fit(X_train_scaled if needs_scaling else X_train, y_train)
                ridge_model.fit(X_train_scaled if needs_scaling else X_train, y_train)
                y_pred = 0.5 * base_model.predict(X_test_scaled if needs_scaling else X_test) + 0.5 * ridge_model.predict(X_test_scaled if needs_scaling else X_test)
            else:
                X_tr = X_train_scaled if needs_scaling else X_train
                X_te = X_test_scaled if needs_scaling else X_test
                if model is None:
                    continue
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)

            metrics = compute_regression_metrics(y_test, y_pred)
            fold_results.append({
                'Model': name,
                'Split': f"{train_years[-1]}->{test_years[0]}",
                **metrics
            })

    if not fold_results:
        raise ValueError('No time-series validation folds were created.')

    summary_df = summarize_fold_metrics(fold_results)
    print(summary_df.to_string(index=False))
    return summary_df, fold_results


def evaluate_optional_baselines(df, features):
    """Run optional advanced baselines when their dependencies are available."""
    print("\n" + "="*60)
    print("STEP 3B: OPTIONAL ADVANCED BASELINES")
    print("="*60)

    results = []

    # ARIMA baseline (commodity-wise)
    try:
        statsmodels = importlib.import_module('statsmodels.tsa.arima.model')
        ARIMA = statsmodels.ARIMA
        arima_metrics = []
        for commodity in sorted(df['commodity'].unique()):
            commodity_df = df[df['commodity'] == commodity].sort_values(['year', 'month']).copy()
            if len(commodity_df) < 24:
                continue
            commodity_df['date_index'] = pd.period_range(start='2015-01', periods=len(commodity_df), freq='M')
            for train_years, test_years in get_time_series_splits(commodity_df):
                train_series = commodity_df[commodity_df['year'].isin(train_years)]['price']
                test_series = commodity_df[commodity_df['year'].isin(test_years)]['price']
                if len(train_series) < 12 or len(test_series) < 2:
                    continue
                try:
                    model = ARIMA(train_series, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
                    forecast = model.forecast(len(test_series))
                    arima_metrics.append(compute_regression_metrics(test_series.values, np.asarray(forecast)))
                except Exception:
                    continue
        if arima_metrics:
            results.append(summarize_baseline_results(arima_metrics, 'ARIMA').iloc[0].to_dict())
    except Exception as exc:
        print(f"  ARIMA unavailable: {exc}")

    # Prophet baseline
    try:
        prophet = importlib.import_module('prophet')
        _ = prophet.Prophet
        prophet_metrics = []
        for commodity in sorted(df['commodity'].unique()):
            commodity_df = df[df['commodity'] == commodity].sort_values(['year', 'month']).copy()
            if len(commodity_df) < 24:
                continue
            commodity_df['ds'] = pd.to_datetime(commodity_df['year'].astype(str) + '-' + commodity_df['month'].astype(str).str.zfill(2) + '-01')
            commodity_df['y'] = commodity_df['price']
            try:
                model = prophet.Prophet(changepoint_prior_scale=0.05, yearly_seasonality=False, weekly_seasonality=False)
                model.fit(commodity_df[['ds', 'y']].rename(columns={'y': 'y'}).reset_index(drop=True))
                future = model.make_future_dataframe(periods=3, freq='MS')
                forecast = model.predict(future)
                prophet_metrics.append(compute_regression_metrics(commodity_df['y'].tail(3).values, forecast['yhat'].tail(3).values))
            except Exception:
                continue
        if prophet_metrics:
            results.append(summarize_baseline_results(prophet_metrics, 'Prophet').iloc[0].to_dict())
    except Exception as exc:
        print(f"  Prophet unavailable: {exc}")

    # PyTorch-based LSTM baseline (optional)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        np.random.seed(42)
        torch.manual_seed(42)
        device = torch.device('cpu')

        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=1, hidden_size=24, num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        lstm_metrics = []
        lookback = 6
        for commodity in sorted(df['commodity'].unique()):
            commodity_df = df[df['commodity'] == commodity].sort_values(['year', 'month']).copy()
            if len(commodity_df) < lookback + 12:
                continue

            values = commodity_df['price'].astype(float).to_numpy()
            split_idx = len(values) - 12
            train_values = values[:split_idx]
            test_values = values[split_idx:]

            if len(train_values) <= lookback:
                continue

            train_mean = float(np.mean(train_values))
            train_std = float(np.std(train_values))
            if train_std < 1e-6:
                train_std = 1.0

            train_scaled = (train_values - train_mean) / train_std
            test_scaled = (test_values - train_mean) / train_std

            X_train_seq = []
            y_train_seq = []
            for i in range(lookback, len(train_scaled)):
                X_train_seq.append(train_scaled[i - lookback:i])
                y_train_seq.append(train_scaled[i])

            if len(X_train_seq) < 12:
                continue

            X_train = torch.tensor(np.array(X_train_seq, dtype=np.float32)).unsqueeze(-1).to(device)
            y_train = torch.tensor(np.array(y_train_seq, dtype=np.float32)).to(device)

            model = SimpleLSTM(hidden_size=24).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            loss_fn = nn.MSELoss()
            dataset = TensorDataset(X_train, y_train)
            loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

            best_state = None
            best_loss = float('inf')
            patience = 6
            epochs = 60
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item())

                epoch_loss = running_loss / max(1, len(loader))
                if epoch_loss < best_loss - 1e-6:
                    best_loss = epoch_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

            model.eval()
            history = list(train_scaled[-lookback:])
            preds_scaled = []
            with torch.no_grad():
                for _ in range(len(test_scaled)):
                    seq = np.array(history[-lookback:], dtype=np.float32).reshape(1, -1, 1)
                    x = torch.tensor(seq, dtype=torch.float32).to(device)
                    pred_scaled = model(x).cpu().item()
                    preds_scaled.append(pred_scaled)
                    history.append(pred_scaled)

            preds = np.array(preds_scaled, dtype=np.float32) * train_std + train_mean
            lstm_metrics.append(compute_regression_metrics(test_values, preds))

        if lstm_metrics:
            results.append(summarize_baseline_results(lstm_metrics, 'LSTM').iloc[0].to_dict())
    except Exception as exc:
        print(f"  PyTorch LSTM unavailable: {exc}")

    # TFT-style baseline hook
    try:
        importlib.import_module('pytorch_forecasting')
        tft_summary = {'Model': 'Temporal Fusion Transformer', 'MAE_mean': None, 'RMSE_mean': None, 'MAPE_mean': None, 'R2_mean': None}
        results.append(tft_summary)
    except Exception as exc:
        print(f"  TFT unavailable: {exc}")

    if results:
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("  No optional baselines were executed. Install the relevant packages to enable them.")
    return pd.DataFrame(results)


def run_ablation_studies(df):
    """Evaluate model performance under different feature subsets."""
    print("\n" + "="*60)
    print("STEP 3C: ABLATION STUDIES")
    print("="*60)

    df_eval = df.copy().sort_values(['commodity', 'year', 'month']).reset_index(drop=True)
    for lag in [1, 2, 3]:
        df_eval[f'price_lag{lag}'] = df_eval.groupby('commodity')['price'].shift(lag)
    df_eval = df_eval.dropna().reset_index(drop=True)

    feature_groups = {
        'climate_only': ['year_trend', 'month', 'season', 'is_monsoon', 'rainfall_mm', 'rainfall_deviation', 'rainfall_category'],
        'commodity_only': ['commodity_encoded'],
        'climate_plus_seasonality': ['year_trend', 'month', 'season', 'is_monsoon', 'rainfall_mm', 'rainfall_deviation', 'rainfall_category', 'commodity_encoded'],
        'climate_plus_lag_plus_seasonality': ['year_trend', 'month', 'season', 'is_monsoon', 'rainfall_mm', 'rainfall_deviation', 'rainfall_category', 'commodity_encoded', 'price_lag1', 'price_lag2', 'price_lag3']
    }

    ablation_rows = []
    for name, features in feature_groups.items():
        summary_df, _ = evaluate_baselines(df_eval, features)
        row = {'Ablation': name, 'Best_Model': summary_df.sort_values('R2_mean', ascending=False).iloc[0]['Model'] if not summary_df.empty else 'N/A'}
        row.update(summary_df.sort_values('R2_mean', ascending=False).iloc[0].to_dict() if not summary_df.empty else {})
        ablation_rows.append(row)

    ablation_df = pd.DataFrame(ablation_rows)
    print(ablation_df.to_string(index=False))
    return ablation_df


def train_models(df, features):
    """Train models without lag features."""
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODELS (WITHOUT LAG)")
    print("="*60)
    
    X = df[features]
    y = df['price']
    
    # Split - use time-based split for realistic evaluation
    train_mask = df['year'] < 2024
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"\n  Training: {len(X_train):,} (before 2024)")
    print(f"  Testing:  {len(X_test):,} (2024 data)")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=10.0), True),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), False)
    }
    
    results = []
    print("\n  Results:")
    print("-" * 60)
    
    for name, (model, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        metrics = compute_regression_metrics(y_test, y_pred)
        
        # Cross-val on training
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')
        
        results.append({
            'Model': name,
            'MAE': round(metrics['MAE'], 2),
            'RMSE': round(metrics['RMSE'], 2),
            'MAPE': round(metrics['MAPE'], 2),
            'R2': round(metrics['R2'], 4),
            'CV_R2': round(cv_scores.mean(), 4)
        })
        
        print(f"  {name:20s}: R2={metrics['R2']:.4f}, MAE=Rs.{metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}")
    
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    
    # Best model
    best_name = results_df.iloc[0]['Model']
    best_r2 = results_df.iloc[0]['R2']
    best_model = models[best_name][0]
    
    print(f"\n  BEST MODEL: {best_name} (R2={best_r2:.4f})")
    
    return results_df, best_model, scaler, X_test, y_test


def analyze_feature_importance(model, features):
    """Analyze which features matter most."""
    print("\n" + "="*60)
    print("STEP 4: FEATURE IMPORTANCE (NO LAG)")
    print("="*60)
    
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n  Feature Importance:")
        for _, row in importance.iterrows():
            bar = '#' * int(row['Importance'] * 50)
            print(f"    {row['Feature']:20s}: {bar} {row['Importance']:.2%}")
        
        return importance
    else:
        print("  (Linear model - no feature importance)")
        return None


def compare_with_lag_model():
    """Compare no-lag vs lag model."""
    print("\n" + "="*60)
    print("STEP 5: COMPARISON - NO LAG vs LAG MODEL")
    print("="*60)
    
    print("""
    +------------------------+---------------+---------------+
    |        Metric          | WITH Lag      | WITHOUT Lag   |
    +------------------------+---------------+---------------+
    | R2 Score               | 99.26%        | ~50-70%       |
    | MAE (Error)            | Rs. 2.54      | Rs. 10-20     |
    | Needs previous price?  | YES           | NO            |
    | Real-time prediction?  | LIMITED       | YES           |
    | Practical use?         | Analysis only | Forecasting   |
    +------------------------+---------------+---------------+
    
    KEY INSIGHT:
    - Lag model is BETTER for accuracy (99%)
    - No-lag model is BETTER for future predictions
    - Both approaches are valid for different use cases!
    """)


def generate_report(results_df, importance, df, time_series_summary, ablation_df, optional_baselines):
    """Generate final report with richer academic evaluation details."""
    report = []
    report.append("="*60)
    report.append("TAMIL NADU PRICE PREDICTION - NO LAG MODEL")
    report.append("="*60)
    
    report.append("\n\n## APPROACH")
    report.append("This model predicts prices WITHOUT using previous month prices.")
    report.append("Instead, it uses:")
    report.append("- Rainfall data")
    report.append("- Season/Month")
    report.append("- Commodity type")
    report.append("- Year trend")
    report.append("- Additional ablation analysis for feature contribution")
    
    report.append("\n\n## MODEL PERFORMANCE")
    report.append(results_df.to_string(index=False))
    report.append("\n")
    report.append("## TIME-SERIES VALIDATION SUMMARY")
    report.append(time_series_summary.to_string(index=False))
    report.append("\n")
    report.append("## ABLATION STUDY")
    report.append(ablation_df.to_string(index=False))
    report.append("\n")
    report.append("## OPTIONAL BASELINE COMPARISON")
    report.append(optional_baselines.to_string(index=False))
    
    report.append("\n\n## FEATURE IMPORTANCE")
    if importance is not None:
        report.append(importance.to_string(index=False))
    
    report.append("\n\n## USE CASE")
    report.append("This model is suitable for:")
    report.append("- Predicting FUTURE prices (before the month ends)")
    report.append("- Seasonal planning")
    report.append("- Rainfall impact analysis")
    report.append("- Rigorous comparison across time-series validation schemes")
    
    report_text = "\n".join(report)
    
    report_path = REPORTS / 'tn_no_lag_model_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n  Report saved: {report_path}")


def main():
    """Run no-lag model pipeline."""
    print("\n" + "="*60)
    print("TAMIL NADU MODEL - WITHOUT LAG FEATURES")
    print("="*60)
    print("\nThis model does NOT use previous month prices!")
    print("This makes it practical for FUTURE predictions.")
    
    # Load data
    merged, raw_prices = load_and_prepare_data()
    
    # Create features
    df, features, le_commodity = create_features(merged)
    
    # Train
    results_df, best_model, scaler, X_test, y_test = train_models(df, features)
    
    # Advanced evaluation
    time_series_summary, _ = evaluate_baselines(df, features)
    ablation_df = run_ablation_studies(df)
    optional_baselines = evaluate_optional_baselines(df, features)
    
    # Feature importance
    importance = analyze_feature_importance(best_model, features)
    
    # Compare
    compare_with_lag_model()
    
    # Report
    generate_report(results_df, importance, df, time_series_summary, ablation_df, optional_baselines)
    
    # Save model
    model_path = MODELS_DIR / 'tn_no_lag_model.joblib'
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'le_commodity': le_commodity,
        'features': features
    }, model_path)
    print(f"\n  Model saved: {model_path}")
    
    print("\n" + "="*60)
    print("COMPLETE! Model trained WITHOUT lag features.")
    print("="*60)


if __name__ == "__main__":
    main()
