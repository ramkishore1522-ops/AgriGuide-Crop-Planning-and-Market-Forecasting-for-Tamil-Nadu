"""
CONFORMAL PREDICTION — Uncertainty Quantification (Novel Contribution)
=======================================================================
Provides distribution-free prediction intervals using split conformal
prediction. Enables risk-aware decision-making for farmers and policymakers.

Outputs:
  - reports/conformal_prediction_results.csv
  - reports/conformal_prediction_report.txt
  - visualizations/figure_prediction_intervals.png
  - visualizations/figure_coverage_analysis.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import sys
import io
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

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


def load_data() -> pd.DataFrame:
    """Load and prepare data (same as per_commodity_pipeline)."""
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


def split_conformal_prediction(
    df_commodity: pd.DataFrame,
    commodity_name: str,
    alpha_levels: list = [0.10, 0.05],  # 90% and 95% confidence
) -> Dict:
    """
    Split Conformal Prediction for one commodity.

    Method:
      1. Split data chronologically into train / calibration / test (60/20/20).
      2. Train model on train set.
      3. Compute nonconformity scores on calibration set: |y - ŷ|.
      4. For each alpha, quantile q = ceil((1-alpha)(1+n_cal)) / n_cal of scores.
      5. Prediction interval on test: [ŷ - q, ŷ + q].
      6. Report empirical coverage and average interval width.
    """
    df = df_commodity.sort_values(['year', 'month']).reset_index(drop=True)

    if len(df) < 30:
        print(f"  [SKIP] {commodity_name}: insufficient data ({len(df)} rows)")
        return {}

    n = len(df)
    n_train = int(n * 0.6)
    n_cal = int(n * 0.2)

    train = df.iloc[:n_train]
    cal = df.iloc[n_train:n_train + n_cal]
    test = df.iloc[n_train + n_cal:]

    if len(test) < 3 or len(cal) < 5:
        print(f"  [SKIP] {commodity_name}: splits too small")
        return {}

    X_train = train[FEATURES].values
    y_train = train['price'].values
    X_cal = cal[FEATURES].values
    y_cal = cal['price'].values
    X_test = test[FEATURES].values
    y_test = test['price'].values

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0,
    )
    model.fit(X_train, y_train)

    # Calibration: compute nonconformity scores
    y_cal_pred = model.predict(X_cal)
    nonconformity_scores = np.abs(y_cal - y_cal_pred)

    # Test predictions
    y_test_pred = model.predict(X_test)

    result = {
        'commodity': commodity_name,
        'n_train': len(train),
        'n_calibration': len(cal),
        'n_test': len(test),
        'point_R2': round(r2_score(y_test, y_test_pred), 4),
        'point_RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2),
        '_y_test': y_test,
        '_y_test_pred': y_test_pred,
        '_test_data': test[['year', 'month']].values,
        '_intervals': {},
    }

    for alpha in alpha_levels:
        confidence = 1 - alpha
        # Conformal quantile
        q_level = np.ceil((1 - alpha) * (len(nonconformity_scores) + 1)) / len(nonconformity_scores)
        q_level = min(q_level, 1.0)
        q_hat = np.quantile(nonconformity_scores, q_level)

        lower = y_test_pred - q_hat
        upper = y_test_pred + q_hat

        # Coverage: fraction of test points inside the interval
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        avg_width = np.mean(upper - lower)

        label = f"{int(confidence * 100)}%"
        result[f'coverage_{label}'] = round(coverage * 100, 1)
        result[f'avg_width_{label}'] = round(avg_width, 2)
        result[f'q_hat_{label}'] = round(q_hat, 2)

        result['_intervals'][label] = {
            'lower': lower,
            'upper': upper,
            'q_hat': q_hat,
        }

        print(f"    {commodity_name} [{label}]: Coverage={coverage * 100:.1f}%, "
              f"Width=₹{avg_width:.2f}, q̂=₹{q_hat:.2f}")

    return result


def run_conformal_all(df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
    """Run conformal prediction for all key commodities."""
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION — UNCERTAINTY QUANTIFICATION")
    print("=" * 70)

    all_results = []
    for commodity in KEY_COMMODITIES:
        df_c = df[df['commodity'] == commodity]
        result = split_conformal_prediction(df_c, commodity)
        if result:
            all_results.append(result)

    # Summary table
    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not k.startswith('_')}
        summary_rows.append(row)

    return all_results, pd.DataFrame(summary_rows)


def visualize_intervals(all_results: list) -> None:
    """Prediction interval visualization."""
    if not all_results:
        return

    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        y_test = result['_y_test']
        y_pred = result['_y_test_pred']
        months = result['_test_data']
        dates = [f"{int(y)}-{int(m):02d}" for y, m in months]
        x = range(len(y_test))

        # Plot 95% interval first (wider, lighter)
        if '95%' in result['_intervals']:
            iv = result['_intervals']['95%']
            ax.fill_between(x, iv['lower'], iv['upper'],
                            alpha=0.2, color='steelblue', label=f"95% CI (±₹{iv['q_hat']:.1f})")

        # Plot 90% interval (narrower, darker)
        if '90%' in result['_intervals']:
            iv = result['_intervals']['90%']
            ax.fill_between(x, iv['lower'], iv['upper'],
                            alpha=0.3, color='steelblue', label=f"90% CI (±₹{iv['q_hat']:.1f})")

        ax.plot(x, y_test, 'ko-', markersize=5, linewidth=2, label='Actual', zorder=5)
        ax.plot(x, y_pred, 'r--', linewidth=1.5, label=f"Predicted (R²={result['point_R2']:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, fontsize=8)
        ax.set_ylabel('Price (₹/kg)')
        ax.set_title(result['commodity'], fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Conformal Prediction Intervals by Commodity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_prediction_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_prediction_intervals.png'}")


def visualize_coverage(summary_df: pd.DataFrame) -> None:
    """Coverage vs target visualization."""
    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Coverage plot ──────────────────────────────────────────────────────
    ax = axes[0]
    commodities = summary_df['commodity'].values

    if 'coverage_90%' in summary_df.columns:
        x = np.arange(len(commodities))
        bars_90 = summary_df['coverage_90%'].values
        bars_95 = summary_df.get('coverage_95%', pd.Series([0] * len(commodities))).values

        w = 0.35
        ax.bar(x - w / 2, bars_90, w, label='90% CI', color='#55A868', alpha=0.85)
        ax.bar(x + w / 2, bars_95, w, label='95% CI', color='#4C72B0', alpha=0.85)
        ax.axhline(y=90, color='green', linestyle='--', linewidth=1, alpha=0.7, label='90% target')
        ax.axhline(y=95, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='95% target')

        ax.set_xticks(x)
        ax.set_xticklabels(commodities, rotation=45)
        ax.set_ylabel('Empirical Coverage (%)')
        ax.set_title('Coverage vs Target', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim([0, 105])

    # ── Interval width plot ────────────────────────────────────────────────
    ax = axes[1]
    if 'avg_width_90%' in summary_df.columns:
        w90 = summary_df['avg_width_90%'].values
        w95 = summary_df.get('avg_width_95%', pd.Series([0] * len(commodities))).values

        w = 0.35
        ax.bar(x - w / 2, w90, w, label='90% CI', color='#55A868', alpha=0.85)
        ax.bar(x + w / 2, w95, w, label='95% CI', color='#4C72B0', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(commodities, rotation=45)
        ax.set_ylabel('Average Interval Width (₹)')
        ax.set_title('Prediction Interval Width', fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle('Conformal Prediction: Coverage & Width Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_coverage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_coverage_analysis.png'}")


def generate_report(summary_df: pd.DataFrame) -> None:
    """Save textual report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CONFORMAL PREDICTION REPORT — UNCERTAINTY QUANTIFICATION")
    lines.append("=" * 70)
    lines.append("\nMethod: Split Conformal Prediction (Vovk et al., 2005)")
    lines.append("  - Distribution-free: no assumptions on error distribution")
    lines.append("  - Guaranteed coverage in exchangeable setting")
    lines.append("  - Practical: calibrated on held-out calibration set\n")
    lines.append("Split: 60% train / 20% calibration / 20% test (chronological)\n")

    lines.append("─" * 70)
    lines.append("RESULTS")
    lines.append("─" * 70)
    display_cols = [c for c in summary_df.columns if not c.startswith('_')]
    lines.append(summary_df[display_cols].to_string(index=False))

    lines.append("\n\nINTERPRETATION:")
    lines.append("  - Coverage ≥ target → valid prediction intervals")
    lines.append("  - Narrower width → more precise uncertainty estimates")
    lines.append("  - Commodities with wider intervals are harder to predict")

    report_path = REPORTS / 'conformal_prediction_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")

    summary_df[[c for c in summary_df.columns if not c.startswith('_')]].to_csv(
        REPORTS / 'conformal_prediction_results.csv', index=False
    )
    print(f"  Saved: {REPORTS / 'conformal_prediction_results.csv'}")


def main() -> None:
    print("=" * 70)
    print("CONFORMAL PREDICTION PIPELINE (Novel Contribution)")
    print("=" * 70)

    df = load_data()
    all_results, summary_df = run_conformal_all(df)

    if not all_results:
        print("\n[ERROR] No results. Check data.")
        return

    visualize_intervals(all_results)
    visualize_coverage(summary_df)
    generate_report(summary_df)

    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
