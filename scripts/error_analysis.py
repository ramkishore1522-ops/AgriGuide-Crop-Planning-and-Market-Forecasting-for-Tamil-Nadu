"""
ERROR ANALYSIS (Publication-Quality)
=====================================
Detailed breakdown of prediction errors by commodity, season, year,
and extreme weather events. Required for any serious ML paper.

Outputs:
  - reports/error_analysis_report.txt
  - visualizations/figure_error_analysis.png
  - visualizations/figure_residual_diagnostics.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import io
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore', category=FutureWarning)
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

FEATURES = [
    'year_trend', 'month', 'month_sin', 'month_cos',
    'season', 'is_monsoon',
    'rainfall_mm', 'rainfall_deviation', 'rainfall_category',
]


def load_and_prepare() -> pd.DataFrame:
    """Load and prepare data identical to per_commodity_pipeline."""
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    prices['date'] = pd.to_datetime(prices['date'])
    tn = prices[prices['state_name'] == 'Tamil Nadu'].copy()
    tn['year'] = tn['date'].dt.year
    tn['month'] = tn['date'].dt.month

    price_monthly = tn.groupby(['commodity', 'year', 'month']).agg(
        price=('price', 'mean')
    ).reset_index()

    # Rainfall
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

    merged = merged.sort_values(['commodity', 'year', 'month']).reset_index(drop=True)
    return merged


SEASON_LABELS = {0: 'Monsoon', 1: 'Post-Monsoon', 2: 'Winter', 3: 'Summer'}


def collect_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each commodity, do a last-fold TimeSeriesSplit prediction
    and return a DataFrame with actual, predicted, residual, and context.
    """
    all_preds = []

    for commodity in KEY_COMMODITIES:
        df_c = df[df['commodity'] == commodity].sort_values(['year', 'month']).reset_index(drop=True)
        if len(df_c) < 20:
            continue

        X = df_c[FEATURES].values
        y = df_c['price'].values

        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]  # Use last fold for error analysis

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        for i, ti in enumerate(test_idx):
            row = df_c.iloc[ti]
            all_preds.append({
                'commodity': commodity,
                'year': int(row['year']),
                'month': int(row['month']),
                'season': SEASON_LABELS.get(int(row['season']), 'Unknown'),
                'is_monsoon': int(row['is_monsoon']),
                'rainfall_mm': float(row['rainfall_mm']),
                'rainfall_deviation': float(row['rainfall_deviation']),
                'actual': float(y[ti]),
                'predicted': float(y_pred[i]),
                'residual': float(y[ti] - y_pred[i]),
                'abs_error': float(abs(y[ti] - y_pred[i])),
                'pct_error': float(abs(y[ti] - y_pred[i]) / max(abs(y[ti]), 1e-6) * 100),
            })

    return pd.DataFrame(all_preds)


def analyze_errors(preds: pd.DataFrame) -> List[str]:
    """Perform comprehensive error analysis and return report lines."""
    report = []
    report.append("=" * 70)
    report.append("ERROR ANALYSIS REPORT (Publication Quality)")
    report.append("=" * 70)

    # ── 1. Error by Commodity ──────────────────────────────────────────────
    report.append("\n\n## 1. ERROR BY COMMODITY")
    report.append("-" * 50)
    by_commodity = preds.groupby('commodity').agg(
        MAE=('abs_error', 'mean'),
        MAPE=('pct_error', 'mean'),
        RMSE=('residual', lambda x: np.sqrt(np.mean(x**2))),
        Mean_Residual=('residual', 'mean'),
        Std_Residual=('residual', 'std'),
        N=('residual', 'count'),
    ).round(3)
    report.append(by_commodity.to_string())

    hardest = by_commodity['MAPE'].idxmax()
    easiest = by_commodity['MAPE'].idxmin()
    report.append(f"\n  Hardest to predict: {hardest} (MAPE={by_commodity.loc[hardest, 'MAPE']:.1f}%)")
    report.append(f"  Easiest to predict: {easiest} (MAPE={by_commodity.loc[easiest, 'MAPE']:.1f}%)")

    # ── 2. Error by Season ─────────────────────────────────────────────────
    report.append("\n\n## 2. ERROR BY SEASON")
    report.append("-" * 50)
    by_season = preds.groupby('season').agg(
        MAE=('abs_error', 'mean'),
        MAPE=('pct_error', 'mean'),
        RMSE=('residual', lambda x: np.sqrt(np.mean(x**2))),
        N=('residual', 'count'),
    ).round(3)
    report.append(by_season.to_string())

    # ── 3. Error by Year ───────────────────────────────────────────────────
    report.append("\n\n## 3. ERROR BY YEAR")
    report.append("-" * 50)
    by_year = preds.groupby('year').agg(
        MAE=('abs_error', 'mean'),
        MAPE=('pct_error', 'mean'),
        N=('residual', 'count'),
    ).round(3)
    report.append(by_year.to_string())

    # ── 4. Extreme Weather Impact ──────────────────────────────────────────
    report.append("\n\n## 4. ERROR DURING EXTREME WEATHER")
    report.append("-" * 50)

    p25 = preds['rainfall_deviation'].quantile(0.25)
    p75 = preds['rainfall_deviation'].quantile(0.75)

    drought = preds[preds['rainfall_deviation'] < p25]
    normal = preds[(preds['rainfall_deviation'] >= p25) & (preds['rainfall_deviation'] <= p75)]
    excess = preds[preds['rainfall_deviation'] > p75]

    for label, subset in [('Drought (<P25)', drought), ('Normal (P25-P75)', normal), ('Excess Rain (>P75)', excess)]:
        if len(subset) > 0:
            report.append(f"\n  {label}: N={len(subset)}, MAE={subset['abs_error'].mean():.2f}, "
                          f"MAPE={subset['pct_error'].mean():.1f}%")

    # ── 5. Residual Normality Test ─────────────────────────────────────────
    report.append("\n\n## 5. RESIDUAL NORMALITY (Shapiro-Wilk)")
    report.append("-" * 50)
    for commodity in preds['commodity'].unique():
        resids = preds[preds['commodity'] == commodity]['residual'].values
        if len(resids) >= 8:
            sample = resids[:min(5000, len(resids))]
            stat, p = sp_stats.shapiro(sample)
            normality = "Normal" if p > 0.05 else "Non-normal"
            report.append(f"  {commodity:10s}: W={stat:.4f}, p={p:.4f} → {normality}")

    # ── 6. Bias Detection ──────────────────────────────────────────────────
    report.append("\n\n## 6. PREDICTION BIAS (One-sample t-test on residuals)")
    report.append("-" * 50)
    for commodity in preds['commodity'].unique():
        resids = preds[preds['commodity'] == commodity]['residual'].values
        if len(resids) >= 5:
            t_stat, p_val = sp_stats.ttest_1samp(resids, 0)
            biased = "BIASED" if p_val < 0.05 else "Unbiased"
            report.append(f"  {commodity:10s}: mean_resid={resids.mean():+.2f}, t={t_stat:.2f}, p={p_val:.4f} → {biased}")

    return report


def plot_error_analysis(preds: pd.DataFrame) -> None:
    """Generate error analysis visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ── 1. Error by Commodity (box plot) ───────────────────────────────────
    ax = axes[0, 0]
    order = preds.groupby('commodity')['pct_error'].median().sort_values(ascending=False).index
    sns.boxplot(data=preds, x='commodity', y='pct_error', order=order, ax=ax, palette='Set2')
    ax.set_xlabel('Commodity')
    ax.set_ylabel('Absolute % Error')
    ax.set_title('Prediction Error Distribution by Commodity', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # ── 2. Error by Season ─────────────────────────────────────────────────
    ax = axes[0, 1]
    sns.boxplot(data=preds, x='season', y='abs_error', ax=ax, palette='coolwarm')
    ax.set_xlabel('Season')
    ax.set_ylabel('Absolute Error (₹)')
    ax.set_title('Prediction Error by Season', fontweight='bold')

    # ── 3. Residual Distribution ───────────────────────────────────────────
    ax = axes[0, 2]
    for commodity in preds['commodity'].unique():
        resids = preds[preds['commodity'] == commodity]['residual']
        ax.hist(resids, bins=20, alpha=0.5, label=commodity, density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Actual − Predicted)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution by Commodity', fontweight='bold')
    ax.legend(fontsize=7)

    # ── 4. Actual vs Predicted scatter ─────────────────────────────────────
    ax = axes[1, 0]
    for commodity in preds['commodity'].unique():
        subset = preds[preds['commodity'] == commodity]
        ax.scatter(subset['actual'], subset['predicted'], s=20, alpha=0.6, label=commodity)
    lims = [preds[['actual', 'predicted']].min().min(), preds[['actual', 'predicted']].max().max()]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Predicted Price (₹)')
    ax.set_title('Actual vs Predicted (All Commodities)', fontweight='bold')
    ax.legend(fontsize=7)

    # ── 5. Error vs Rainfall Deviation ─────────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(preds['rainfall_deviation'], preds['abs_error'], s=15, alpha=0.5, c='steelblue')
    ax.set_xlabel('Rainfall Deviation (%)')
    ax.set_ylabel('Absolute Error (₹)')
    ax.set_title('Error vs Rainfall Deviation', fontweight='bold')
    # Trendline
    if len(preds) > 10:
        z = np.polyfit(preds['rainfall_deviation'].dropna(), preds['abs_error'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(preds['rainfall_deviation'].min(), preds['rainfall_deviation'].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
        ax.legend()

    # ── 6. Error over time ─────────────────────────────────────────────────
    ax = axes[1, 2]
    time_error = preds.groupby(['year', 'month'])['abs_error'].mean().reset_index()
    time_error['date'] = pd.to_datetime(time_error['year'].astype(str) + '-' + time_error['month'].astype(str) + '-01')
    ax.plot(time_error['date'], time_error['abs_error'], marker='o', markersize=3, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Absolute Error (₹)')
    ax.set_title('Prediction Error Over Time', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Comprehensive Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_error_analysis.png'}")


def plot_residual_diagnostics(preds: pd.DataFrame) -> None:
    """Q-Q plot and residual vs fitted diagnostics."""
    commodities = preds['commodity'].unique()
    n = len(commodities)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i, commodity in enumerate(commodities):
        subset = preds[preds['commodity'] == commodity]

        # Q-Q plot
        ax = axes[0, i]
        sp_stats.probplot(subset['residual'], dist='norm', plot=ax)
        ax.set_title(f'{commodity}\nQ-Q Plot', fontsize=10, fontweight='bold')

        # Residual vs Fitted
        ax = axes[1, i]
        ax.scatter(subset['predicted'], subset['residual'], s=15, alpha=0.6)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(f'{commodity}\nResidual vs Fitted', fontsize=10, fontweight='bold')

    plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_residual_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_residual_diagnostics.png'}")


def main() -> None:
    print("=" * 70)
    print("ERROR ANALYSIS PIPELINE")
    print("=" * 70)

    df = load_and_prepare()
    print(f"  Loaded {len(df):,} records")

    preds = collect_predictions(df)
    print(f"  Collected {len(preds):,} test predictions")

    if preds.empty:
        print("  [ERROR] No predictions collected. Check data.")
        return

    # Analysis
    report_lines = analyze_errors(preds)

    # Save report
    report_path = REPORTS / 'error_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_path}")

    # Save predictions
    preds.to_csv(REPORTS / 'error_predictions.csv', index=False)

    # Visualizations
    plot_error_analysis(preds)
    plot_residual_diagnostics(preds)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
