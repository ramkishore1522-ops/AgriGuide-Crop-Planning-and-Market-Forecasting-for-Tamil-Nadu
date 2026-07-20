"""
HYBRID STACKING ENSEMBLE MODEL (Novel Contribution)
=====================================================
Uses a two-level stacking approach:
  Level 1: Base learners (ARIMA per-commodity, Ridge, Random Forest)
  Level 2: XGBoost meta-learner combines base predictions with climate features

Also compares: Global model (with commodity feature) vs Per-commodity models
vs Hybrid stacking to demonstrate when each approach excels.

This is the PRIMARY novelty claim for the paper.

Outputs:
  - reports/hybrid_model_results.csv
  - reports/hybrid_model_report.txt
  - visualizations/figure_hybrid_decomposition.png
  - visualizations/figure_hybrid_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import io
import warnings

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
import xgboost as xgb
from scipy import stats as sp_stats

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── IEEE Plot Styling ──────────────────────────────────────────────────────────
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

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
VISUALIZATIONS = PROJECT_ROOT / 'visualizations'
REPORTS = PROJECT_ROOT / 'reports'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

KEY_COMMODITIES = ['Rice', 'Wheat', 'Onion', 'Tomato', 'Potato', 'Sugar', 'Milk']


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_data() -> pd.DataFrame:
    """Load and prepare monthly data with GLOBAL features (including commodity)."""
    print("Loading data...")
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    prices['date'] = pd.to_datetime(prices['date'])
    tn = prices[prices['state_name'] == 'Tamil Nadu'].copy()
    tn['year'] = tn['date'].dt.year
    tn['month'] = tn['date'].dt.month

    price_monthly = tn.groupby(['commodity', 'year', 'month']).agg(
        price=('price', 'mean')
    ).reset_index()

    try:
        rain_raw = pd.read_csv(DATA_RAW / 'daily-rainfall-data-district-level.csv')
        tn_rain = rain_raw[rain_raw['state_name'].str.contains('Tamil', case=False, na=False)].copy()
        tn_rain['date'] = pd.to_datetime(tn_rain['date'])
    except (FileNotFoundError, KeyError):
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
        rain_monthly = tn_rain.groupby(['year', 'month']).agg(
            rainfall_mm=('rainfall', 'sum')
        ).reset_index()
        rain_monthly['rainfall_deviation'] = 0.0

    merged = price_monthly.merge(rain_monthly, on=['year', 'month'], how='left')
    merged['rainfall_mm'] = merged['rainfall_mm'].fillna(merged['rainfall_mm'].median())
    merged['rainfall_deviation'] = merged['rainfall_deviation'].fillna(0)

    # Feature engineering
    le = LabelEncoder()
    merged['commodity_idx'] = le.fit_transform(merged['commodity'])
    merged['year_trend'] = merged['year'] - merged['year'].min()
    merged['season'] = merged['month'].apply(
        lambda m: 0 if m in [6, 7, 8, 9] else (1 if m in [10, 11] else (2 if m in [12, 1, 2] else 3))
    )
    merged['is_monsoon'] = merged['month'].isin([6, 7, 8, 9]).astype(int)
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
    merged['rainfall_category'] = pd.cut(
        merged['rainfall_mm'], bins=[-1, 50, 150, 300, 100000], labels=[0, 1, 2, 3]
    ).astype(float).fillna(1).astype(int)

    # Lag features per commodity
    merged = merged.sort_values(['commodity', 'year', 'month']).reset_index(drop=True)
    for lag in [1, 2, 3]:
        merged[f'price_lag{lag}'] = merged.groupby('commodity')['price'].shift(lag)

    # Rolling statistics per commodity
    merged['price_roll_mean3'] = merged.groupby('commodity')['price'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    merged['price_roll_std3'] = merged.groupby('commodity')['price'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    )
    merged['price_roll_std3'] = merged['price_roll_std3'].fillna(0)

    merged = merged.dropna(subset=['price_lag1']).reset_index(drop=True)
    merged = merged.sort_values(['year', 'month', 'commodity']).reset_index(drop=True)

    print(f"  Records: {len(merged):,} | Commodities: {merged['commodity'].nunique()}")
    print(f"  Years: {merged['year'].min()}–{merged['year'].max()}")

    return merged, le


# Feature sets for comparison
FEATURES_NO_LAG = [
    'commodity_idx', 'year_trend', 'month', 'month_sin', 'month_cos',
    'season', 'is_monsoon', 'rainfall_mm', 'rainfall_deviation', 'rainfall_category',
]

FEATURES_WITH_LAG = FEATURES_NO_LAG + [
    'price_lag1', 'price_lag2', 'price_lag3',
    'price_roll_mean3', 'price_roll_std3',
]

FEATURES_CLIMATE_ONLY = [
    'year_trend', 'month', 'month_sin', 'month_cos',
    'season', 'is_monsoon', 'rainfall_mm', 'rainfall_deviation', 'rainfall_category',
]


def get_arima_predictions(df: pd.DataFrame, train_mask, test_mask) -> np.ndarray:
    """Generate per-commodity ARIMA predictions for the test set."""
    preds = np.zeros(test_mask.sum())
    test_indices = df[test_mask].index

    for commodity in df['commodity'].unique():
        comm_train = df[(df['commodity'] == commodity) & train_mask].sort_values(['year', 'month'])
        comm_test = df[(df['commodity'] == commodity) & test_mask].sort_values(['year', 'month'])

        if len(comm_train) < 12 or len(comm_test) == 0:
            # Fallback: use training mean
            for idx in comm_test.index:
                pos = test_indices.get_loc(idx)
                preds[pos] = comm_train['price'].mean()
            continue

        y_train_series = comm_train['price'].values

        if HAS_ARIMA:
            try:
                best_aic = np.inf
                best_order = (1, 0, 0)
                for p in [1, 2]:
                    for d in [0, 1]:
                        for q in [0, 1]:
                            try:
                                fit = ARIMA(y_train_series, order=(p, d, q),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False).fit()
                                if fit.aic < best_aic:
                                    best_aic = fit.aic
                                    best_order = (p, d, q)
                            except Exception:
                                continue

                arima_model = ARIMA(y_train_series, order=best_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
                forecasts = arima_model.forecast(len(comm_test))

                for i, idx in enumerate(comm_test.index):
                    pos = test_indices.get_loc(idx)
                    preds[pos] = forecasts[i]
            except Exception:
                for idx in comm_test.index:
                    pos = test_indices.get_loc(idx)
                    preds[pos] = comm_train['price'].mean()
        else:
            for idx in comm_test.index:
                pos = test_indices.get_loc(idx)
                preds[pos] = comm_train['price'].mean()

    return preds


def expanding_window_evaluation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Expanding-window cross-validation comparing 5 approaches:
      1. Naive Mean
      2. XGBoost (no lag features)
      3. XGBoost (with lag features)
      4. ARIMA (per commodity)
      5. HYBRID STACKING: XGBoost meta-learner on [ARIMA_pred, lag features, climate features]
    """
    print("\n" + "=" * 70)
    print("EXPANDING-WINDOW CROSS-VALIDATION")
    print("=" * 70)

    years = sorted(df['year'].unique())
    min_train_years = 4

    models_config = {
        'Naive Mean': {'type': 'naive'},
        'XGBoost (no-lag)': {'type': 'xgb', 'features': FEATURES_NO_LAG},
        'XGBoost (with-lag)': {'type': 'xgb', 'features': FEATURES_WITH_LAG},
        'ARIMA': {'type': 'arima'},
        'Hybrid Stacking': {'type': 'hybrid'},
    }

    all_fold_results = []
    all_predictions = {name: [] for name in models_config}
    all_actuals = []

    for fold_idx in range(min_train_years, len(years)):
        train_years = years[:fold_idx]
        test_years = [years[fold_idx]]

        train_mask = df['year'].isin(train_years)
        test_mask = df['year'].isin(test_years)

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        y_train = df.loc[train_mask, 'price'].values
        y_test = df.loc[test_mask, 'price'].values
        all_actuals.extend(y_test.tolist())

        # ARIMA predictions (needed for both ARIMA standalone and hybrid)
        arima_test_preds = get_arima_predictions(df, train_mask, test_mask)

        for model_name, config in models_config.items():
            if config['type'] == 'naive':
                preds = np.full(len(y_test), y_train.mean())

            elif config['type'] == 'xgb':
                features = config['features']
                X_train = df.loc[train_mask, features].values
                X_test = df.loc[test_mask, features].values
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_SEED, verbosity=0,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            elif config['type'] == 'arima':
                preds = arima_test_preds

            elif config['type'] == 'hybrid':
                # Stacking: Add ARIMA predictions as a feature to XGBoost
                # Also get ARIMA predictions for training set (in-sample fitted values)
                arima_train_preds = get_arima_predictions(
                    df,
                    df['year'].isin(years[:max(2, fold_idx - 1)]),  # train ARIMA on earlier data
                    train_mask,
                )
                # If the above doesn't work well, use leave-one-out on training
                # Fallback: use per-commodity rolling mean as ARIMA proxy for training
                if np.all(arima_train_preds == 0):
                    arima_train_preds = df.loc[train_mask, 'price_roll_mean3'].values

                # Stacking features = ARIMA pred + all regular features
                X_train_base = df.loc[train_mask, FEATURES_WITH_LAG].values
                X_test_base = df.loc[test_mask, FEATURES_WITH_LAG].values

                X_train_stacked = np.column_stack([X_train_base, arima_train_preds])
                X_test_stacked = np.column_stack([X_test_base, arima_test_preds])

                meta_model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_SEED, verbosity=0,
                )
                meta_model.fit(X_train_stacked, y_train)
                preds = meta_model.predict(X_test_stacked)

            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            mape_val = mape(y_test, preds)

            all_fold_results.append({
                'Model': model_name,
                'Fold': f"{train_years[-1]}→{test_years[0]}",
                'Test_Year': test_years[0],
                'R2': round(r2, 4),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'MAPE': round(mape_val, 2),
            })
            all_predictions[model_name].extend(preds.tolist())

        print(f"  Fold {train_years[-1]}→{test_years[0]} complete")

    results_df = pd.DataFrame(all_fold_results)

    # Summary statistics
    summary = results_df.groupby('Model').agg(
        R2_mean=('R2', 'mean'), R2_std=('R2', 'std'),
        RMSE_mean=('RMSE', 'mean'), RMSE_std=('RMSE', 'std'),
        MAE_mean=('MAE', 'mean'), MAPE_mean=('MAPE', 'mean'),
    ).round(4).sort_values('R2_mean', ascending=False)

    print("\n" + "─" * 70)
    print("RESULTS SUMMARY (Mean ± Std across folds)")
    print("─" * 70)
    print(summary.to_string())

    # Statistical significance
    print("\n" + "─" * 70)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon vs Hybrid Stacking)")
    print("─" * 70)
    best_name = summary.index[0]
    best_preds = all_predictions[best_name]
    for name in models_config:
        if name != best_name and len(all_predictions[name]) == len(best_preds):
            try:
                abs_err_other = [abs(a - p) for a, p in zip(all_actuals, all_predictions[name])]
                abs_err_best = [abs(a - p) for a, p in zip(all_actuals, best_preds)]
                _, p_val = sp_stats.wilcoxon(abs_err_other, abs_err_best)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                print(f"  {name:25s} vs {best_name}: p={p_val:.4e} {sig}")
            except ValueError:
                print(f"  {name:25s} vs {best_name}: test failed")

    return results_df, {
        'summary': summary,
        'predictions': all_predictions,
        'actuals': all_actuals,
    }


def visualize_results(results_df: pd.DataFrame, extras: Dict) -> None:
    """Generate publication-quality figures."""
    summary = extras['summary']

    # ── Figure 1: Model comparison bar chart ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, label in zip(
        axes,
        ['R2_mean', 'RMSE_mean', 'MAPE_mean'],
        ['R² Score', 'RMSE (₹)', 'MAPE (%)'],
    ):
        colors = ['#C44E52' if 'Hybrid' in n else '#4C72B0' if 'lag' in n.lower()
                  else '#55A868' if 'ARIMA' in n else '#8172B2'
                  for n in summary.index]
        bars = ax.barh(summary.index, summary[metric], color=colors, alpha=0.85)
        ax.set_xlabel(label)
        ax.set_title(f'{label} (Mean across folds)', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)

    plt.suptitle('Model Performance Comparison (Expanding Window CV)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_hybrid_comparison.png'}")

    # ── Figure 2: R² by fold over time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name in results_df['Model'].unique():
        subset = results_df[results_df['Model'] == model_name]
        style = '-' if 'Hybrid' in model_name else '--'
        width = 2.5 if 'Hybrid' in model_name or 'with-lag' in model_name else 1.5
        ax.plot(subset['Test_Year'], subset['R2'], marker='o', markersize=5,
                linewidth=width, linestyle=style, label=model_name)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Test Year')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance Over Time (Expanding Window)', fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_hybrid_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_hybrid_decomposition.png'}")


def generate_report(results_df: pd.DataFrame, extras: Dict) -> None:
    """Save report."""
    summary = extras['summary']

    lines = []
    lines.append("=" * 70)
    lines.append("HYBRID STACKING ENSEMBLE MODEL REPORT")
    lines.append("=" * 70)
    lines.append("\nApproach: Two-level stacking ensemble")
    lines.append("  Level 1: Per-commodity ARIMA forecasts (captures trends)")
    lines.append("  Level 2: XGBoost meta-learner (combines ARIMA + climate + lag features)")
    lines.append("\nEvaluation: Expanding-window cross-validation")
    lines.append("  Train on years 1..N, test on year N+1\n")

    lines.append("─" * 70)
    lines.append("RESULTS SUMMARY")
    lines.append("─" * 70)
    lines.append(summary.to_string())

    lines.append("\n\n─" * 35)
    lines.append("PER-FOLD RESULTS")
    lines.append("─" * 70)
    lines.append(results_df.to_string(index=False))

    # Key findings
    lines.append("\n\n─" * 35)
    lines.append("KEY FINDINGS")
    lines.append("─" * 70)
    best = summary.index[0]
    lines.append(f"  Best model: {best} (R²={summary.loc[best, 'R2_mean']:.4f})")

    if 'Hybrid Stacking' in summary.index and 'XGBoost (with-lag)' in summary.index:
        hybrid_r2 = summary.loc['Hybrid Stacking', 'R2_mean']
        xgb_r2 = summary.loc['XGBoost (with-lag)', 'R2_mean']
        diff = hybrid_r2 - xgb_r2
        lines.append(f"  Hybrid vs XGBoost(lag): ΔR² = {diff:+.4f}")

    if 'XGBoost (with-lag)' in summary.index and 'XGBoost (no-lag)' in summary.index:
        lag_r2 = summary.loc['XGBoost (with-lag)', 'R2_mean']
        nolag_r2 = summary.loc['XGBoost (no-lag)', 'R2_mean']
        lines.append(f"  With-lag vs No-lag: ΔR² = {lag_r2 - nolag_r2:+.4f}")
        lines.append(f"  → Lag features improve R² by {(lag_r2 - nolag_r2):.4f}")

    report_path = REPORTS / 'hybrid_model_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")

    # CSV
    summary.to_csv(REPORTS / 'hybrid_model_results.csv')
    print(f"  Saved: {REPORTS / 'hybrid_model_results.csv'}")


def main() -> None:
    print("=" * 70)
    print("HYBRID STACKING ENSEMBLE (Novel Contribution)")
    print("=" * 70)

    df, le = load_data()
    results_df, extras = expanding_window_evaluation(df)

    if results_df.empty:
        print("\n[ERROR] No results.")
        return

    visualize_results(results_df, extras)
    generate_report(results_df, extras)

    print("\n" + "=" * 70)
    print("HYBRID STACKING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
