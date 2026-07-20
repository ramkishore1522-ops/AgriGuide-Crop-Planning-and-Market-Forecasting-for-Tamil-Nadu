"""
GENERATE PAPER TABLES (Publication-Quality LaTeX)
==================================================
Converts all generated CSV reports into properly formatted LaTeX tables
for direct inclusion in an IEEE or MDPI journal publication.

Outputs:
  - reports/latex_tables.tex (complete, compilable LaTeX table set)
"""

import pandas as pd
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS = PROJECT_ROOT / 'reports'


def safe_load_csv(filename: str) -> pd.DataFrame:
    """Load CSV from reports directory, return empty DataFrame if not found."""
    path = REPORTS / filename
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [SKIP] {filename} not found")
        return pd.DataFrame()


def df_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str = None,
) -> str:
    """Convert DataFrame to a properly formatted LaTeX table string."""
    if df.empty:
        return f"% TABLE SKIPPED: {label} (no data)\n"

    if column_format is None:
        column_format = 'l' + 'c' * (len(df.columns) - 1)

    latex_body = df.to_latex(index=False, column_format=column_format, escape=True)

    return (
        f"% {caption}\n"
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\small\n"
        f"{latex_body}"
        f"\\end{{table}}\n\n"
    )


def generate_all_tables() -> None:
    """Generate all LaTeX tables for the paper."""
    print("=" * 70)
    print("GENERATING LATEX TABLES FOR PAPER")
    print("=" * 70)

    latex_output = []
    latex_output.append("% ============================================")
    latex_output.append("% AUTO-GENERATED LATEX TABLES FOR PUBLICATION")
    latex_output.append("% ============================================\n")

    # ── Table 1: Dataset Description ───────────────────────────────────────
    dataset_info = pd.DataFrame([
        {'Source': 'Retail Prices', 'Records': '~2.1M', 'Period': '2015–2024',
         'Granularity': 'Daily, State/District', 'Variables': 'commodity, price, state'},
        {'Source': 'Rainfall (State)', 'Records': '~300K', 'Period': '2015–2024',
         'Granularity': 'Daily, State', 'Variables': 'actual, deviation'},
        {'Source': 'Rainfall (District)', 'Records': '~5.6M', 'Period': '2015–2024',
         'Granularity': 'Daily, District', 'Variables': 'actual, deviation'},
        {'Source': 'Cost of Cultivation', 'Records': '~30K', 'Period': '2012–2023',
         'Granularity': 'Annual, State', 'Variables': 'cost, revenue, yield'},
        {'Source': 'MSP', 'Records': '~600', 'Period': '1966–2024',
         'Granularity': 'Annual', 'Variables': 'crop, min\\_support\\_price'},
        {'Source': 'Groundwater', 'Records': '~1.4M', 'Period': '2015–2024',
         'Granularity': 'Quarterly, District', 'Variables': 'water\\_level'},
        {'Source': 'Vulnerability Index', 'Records': '~36', 'Period': '2020',
         'Granularity': 'State', 'Variables': 'climate\\_vul\\_in'},
    ])
    latex_output.append(df_to_latex_table(
        dataset_info,
        'Dataset Description and Sources',
        'tab:dataset',
        'lrllp{5cm}',
    ))

    # ── Table 2: Model Comparison (Publication Pipeline) ───────────────────
    model_comp = safe_load_csv('table_model_comparison.csv')
    if not model_comp.empty:
        latex_output.append(df_to_latex_table(
            model_comp,
            'Global Model Performance Comparison (5-Fold TimeSeries CV)',
            'tab:model_comp',
        ))

    # ── Table 3: Per-Commodity Results ─────────────────────────────────────
    per_commodity = safe_load_csv('per_commodity_results.csv')
    if not per_commodity.empty:
        # Show only best model per commodity for the main table
        best_per = per_commodity.loc[
            per_commodity.groupby('Commodity')['R2_mean'].idxmax()
        ][['Commodity', 'Model', 'R2_mean', 'R2_std', 'RMSE_mean', 'MAE_mean', 'MAPE_mean']].copy()
        best_per.columns = ['Commodity', 'Best Model', 'R²', 'R² (std)', 'RMSE', 'MAE', 'MAPE (\\%)']
        latex_output.append(df_to_latex_table(
            best_per,
            'Per-Commodity Model Performance (Best Model, 5-Fold TimeSeriesSplit)',
            'tab:per_commodity',
        ))

    # ── Table 4: Hybrid Model Results ──────────────────────────────────────
    hybrid = safe_load_csv('hybrid_model_results.csv')
    if not hybrid.empty:
        cols = [c for c in hybrid.columns if not c.startswith('_')]
        display_cols = ['commodity']
        for prefix in ['ARIMA', 'XGBoost', 'Hybrid']:
            for metric in ['R2', 'RMSE', 'MAPE']:
                col = f'{prefix}_{metric}'
                if col in cols:
                    display_cols.append(col)
        available = [c for c in display_cols if c in hybrid.columns]
        latex_output.append(df_to_latex_table(
            hybrid[available],
            'Hybrid ARIMA+XGBoost vs Standalone Models',
            'tab:hybrid',
        ))

    # ── Table 5: Granger Causality ─────────────────────────────────────────
    granger = safe_load_csv('granger_causality_results.csv')
    if not granger.empty:
        sig_only = granger[granger['Significant (p<0.05)'] == 'Yes']
        if not sig_only.empty:
            display = sig_only[['Commodity', 'Climate Variable', 'Lag (months)',
                                'F-statistic', 'p-value']].copy()
            latex_output.append(df_to_latex_table(
                display,
                'Significant Granger Causality Results (p $<$ 0.05)',
                'tab:granger',
            ))

    # ── Table 6: Conformal Prediction ──────────────────────────────────────
    conformal = safe_load_csv('conformal_prediction_results.csv')
    if not conformal.empty:
        display_cols = [c for c in conformal.columns if not c.startswith('_')]
        latex_output.append(df_to_latex_table(
            conformal[display_cols],
            'Conformal Prediction: Coverage and Interval Width',
            'tab:conformal',
        ))

    # ── Table 7: Price-Climate Correlations ────────────────────────────────
    corr = safe_load_csv('table_price_climate_correlation.csv')
    if not corr.empty:
        latex_output.append(df_to_latex_table(
            corr,
            'Statistical Significance of Price-Rainfall Correlations',
            'tab:correlation',
        ))

    # ── Write output ───────────────────────────────────────────────────────
    output_path = REPORTS / 'latex_tables.tex'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_output))

    print(f"\n  LaTeX tables generated at: {output_path}")
    print(f"  Total tables: {sum(1 for line in latex_output if '\\begin{table}' in line)}")


if __name__ == '__main__':
    generate_all_tables()
