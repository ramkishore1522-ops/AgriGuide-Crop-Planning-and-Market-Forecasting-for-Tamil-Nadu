"""
GRANGER CAUSALITY ANALYSIS (Novel Contribution)
=================================================
Tests whether climate variables (rainfall, deviation) temporally
Granger-cause commodity prices. This directly addresses the weak
correlation finding by testing LAGGED causal effects.

Outputs:
  - reports/granger_causality_results.csv
  - reports/granger_causality_report.txt
  - visualizations/figure_granger_causality.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys
import io
import warnings

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── IEEE Plot Styling ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_QUALITY = PROJECT_ROOT / "data" / "quality_checked"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
VISUALIZATIONS = PROJECT_ROOT / "visualizations"
REPORTS = PROJECT_ROOT / "reports"

KEY_COMMODITIES = ["Rice", "Wheat", "Onion", "Tomato", "Potato", "Sugar", "Milk"]
MAX_LAGS = 6  # Test lags 1–6 months


def load_data() -> pd.DataFrame:
    """Load and prepare monthly data."""
    prices = pd.read_csv(DATA_QUALITY / "retail_prices_quality.csv")
    prices["date"] = pd.to_datetime(prices["date"])
    tn = prices[prices["state_name"] == "Tamil Nadu"].copy()
    tn["year"] = tn["date"].dt.year
    tn["month"] = tn["date"].dt.month

    price_monthly = (
        tn.groupby(["commodity", "year", "month"])
        .agg(price=("price", "mean"))
        .reset_index()
    )

    try:
        rain_raw = pd.read_csv(DATA_RAW / "daily-rainfall-data-district-level.csv")
        tn_rain = rain_raw[
            rain_raw["state_name"].str.contains("Tamil", case=False, na=False)
        ].copy()
        tn_rain["date"] = pd.to_datetime(tn_rain["date"])
    except (FileNotFoundError, KeyError):
        rain_raw = pd.read_csv(DATA_QUALITY / "rainfall_state_quality.csv")
        tn_rain = rain_raw[rain_raw["state_name"] == "Tamil Nadu"].copy()
        tn_rain["date"] = pd.to_datetime(tn_rain["date"])

    tn_rain["year"] = tn_rain["date"].dt.year
    tn_rain["month"] = tn_rain["date"].dt.month

    if "actual" in tn_rain.columns:
        rain_monthly = (
            tn_rain.groupby(["year", "month"])
            .agg(
                rainfall_mm=("actual", "sum"), rainfall_deviation=("deviation", "mean")
            )
            .reset_index()
        )
    else:
        rain_monthly = (
            tn_rain.groupby(["year", "month"])
            .agg(rainfall_mm=("rainfall", "sum"))
            .reset_index()
        )
        rain_monthly["rainfall_deviation"] = 0.0

    merged = price_monthly.merge(rain_monthly, on=["year", "month"], how="left")
    merged["rainfall_mm"] = merged["rainfall_mm"].fillna(merged["rainfall_mm"].median())
    merged["rainfall_deviation"] = merged["rainfall_deviation"].fillna(0)
    merged = merged.sort_values(["commodity", "year", "month"]).reset_index(drop=True)

    return merged


def check_stationarity(series: np.ndarray, name: str) -> Dict:
    """Run Augmented Dickey-Fuller test."""
    if not HAS_STATSMODELS:
        return {
            "name": name,
            "stationary": "UNKNOWN",
            "adf_stat": None,
            "p_value": None,
        }

    result = adfuller(series, autolag="AIC")
    return {
        "name": name,
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "stationary": "Yes" if result[1] < 0.05 else "No",
    }


def granger_test_commodity(
    df_commodity: pd.DataFrame,
    commodity_name: str,
    climate_var: str = "rainfall_mm",
    max_lags: int = MAX_LAGS,
) -> List[Dict]:
    """
    Test if climate_var Granger-causes price for one commodity.

    Returns list of dicts with results per lag.
    """
    df = df_commodity.sort_values(["year", "month"]).reset_index(drop=True)

    if len(df) < max_lags + 15:
        print(f"  [SKIP] {commodity_name}: insufficient data ({len(df)} rows)")
        return []

    # Prepare the bivariate series: [price, climate_var]
    # Granger test needs: column 0 = effect (price), column 1 = cause (climate)
    data = df[["price", climate_var]].dropna().values

    if len(data) < max_lags + 10:
        return []

    results = []
    try:
        gc_result = grangercausalitytests(data, maxlag=max_lags, verbose=False)

        for lag in range(1, max_lags + 1):
            test_result = gc_result[lag]
            # Extract F-test results (ssr_ftest)
            f_test = test_result[0]["ssr_ftest"]
            f_stat = f_test[0]
            p_value = f_test[1]

            results.append(
                {
                    "Commodity": commodity_name,
                    "Climate Variable": climate_var,
                    "Lag (months)": lag,
                    "F-statistic": round(f_stat, 4),
                    "p-value": round(p_value, 4),
                    "Significant (p<0.05)": "Yes" if p_value < 0.05 else "No",
                    "Significant (p<0.01)": "Yes" if p_value < 0.01 else "No",
                }
            )

    except Exception as e:
        print(f"    Granger test failed for {commodity_name} ({climate_var}): {e}")

    return results


def run_granger_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run Granger causality for all commodities and climate variables."""
    print("\n" + "=" * 70)
    print("GRANGER CAUSALITY ANALYSIS")
    print("=" * 70)

    if not HAS_STATSMODELS:
        print(
            "  [ERROR] statsmodels is required. Install with: pip install statsmodels"
        )
        return pd.DataFrame()

    all_results = []

    for commodity in KEY_COMMODITIES:
        df_c = df[df["commodity"] == commodity]
        print(f"\n  ── {commodity} ──")

        # Stationarity check
        if len(df_c) >= 20:
            price_stat = check_stationarity(df_c["price"].values, f"{commodity}_price")
            rain_stat = check_stationarity(
                df_c["rainfall_mm"].values, f"{commodity}_rainfall"
            )
            print(
                f"    Price stationary: {price_stat['stationary']} (p={price_stat['p_value']})"
            )
            print(
                f"    Rainfall stationary: {rain_stat['stationary']} (p={rain_stat['p_value']})"
            )

        # Test rainfall → price
        results_rain = granger_test_commodity(df_c, commodity, "rainfall_mm")
        all_results.extend(results_rain)

        # Test deviation → price
        results_dev = granger_test_commodity(df_c, commodity, "rainfall_deviation")
        all_results.extend(results_dev)

        # Report best lag
        sig_results = [r for r in results_rain if r["Significant (p<0.05)"] == "Yes"]
        if sig_results:
            best = min(sig_results, key=lambda x: x["p-value"])
            print(
                f"    ✓ Rainfall → Price: SIGNIFICANT at lag {best['Lag (months)']} "
                f"(F={best['F-statistic']:.2f}, p={best['p-value']:.4f})"
            )
        else:
            print("    ✗ Rainfall → Price: NOT significant at any lag")

        sig_dev = [r for r in results_dev if r["Significant (p<0.05)"] == "Yes"]
        if sig_dev:
            best = min(sig_dev, key=lambda x: x["p-value"])
            print(
                f"    ✓ Deviation → Price: SIGNIFICANT at lag {best['Lag (months)']} "
                f"(F={best['F-statistic']:.2f}, p={best['p-value']:.4f})"
            )
        else:
            print("    ✗ Deviation → Price: NOT significant at any lag")

    return pd.DataFrame(all_results)


def visualize_granger(results_df: pd.DataFrame) -> None:
    """Heatmap of Granger causality p-values."""
    if results_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, climate_var, title in zip(
        axes,
        ["rainfall_mm", "rainfall_deviation"],
        ["Rainfall → Price", "Rainfall Deviation → Price"],
    ):
        subset = results_df[results_df["Climate Variable"] == climate_var]
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        pivot = subset.pivot(
            index="Commodity", columns="Lag (months)", values="p-value"
        )

        # Create significance mask
        sig_mask = pivot > 0.05  # Non-significant cells

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=ax,
            vmin=0,
            vmax=0.1,
            cbar_kws={"label": "p-value"},
            linewidths=0.5,
        )

        # Add significance stars
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if val < 0.01:
                    ax.text(
                        j + 0.5,
                        i + 0.15,
                        "***",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                    )
                elif val < 0.05:
                    ax.text(
                        j + 0.5,
                        i + 0.15,
                        "**",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                    )

        ax.set_title(f"Granger Causality: {title}", fontweight="bold")
        ax.set_xlabel("Lag (months)")

    plt.suptitle(
        "Granger Causality Test Results (p-values)\n"
        "*** p<0.01, ** p<0.05; green=significant",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        VISUALIZATIONS / "figure_granger_causality.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_granger_causality.png'}")


def generate_report(results_df: pd.DataFrame) -> None:
    """Save textual report."""
    lines = []
    lines.append("=" * 70)
    lines.append("GRANGER CAUSALITY ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(
        "\nTest: Does climate (rainfall/deviation) Granger-cause commodity prices?"
    )
    lines.append("Method: Standard Granger causality F-test (statsmodels)")
    lines.append(f"Max lags tested: {MAX_LAGS}")
    lines.append("Null hypothesis: Climate does NOT Granger-cause price\n")

    if results_df.empty:
        lines.append("No results generated.")
    else:
        # Summary: which commodities have significant causal links
        lines.append("─" * 70)
        lines.append("SUMMARY: SIGNIFICANT CAUSAL LINKS (p < 0.05)")
        lines.append("─" * 70)

        sig = results_df[results_df["Significant (p<0.05)"] == "Yes"]
        if sig.empty:
            lines.append("  No statistically significant Granger causal links found.")
        else:
            for _, row in sig.iterrows():
                lines.append(
                    f"  {row['Climate Variable']:20s} → {row['Commodity']:10s} "
                    f"at lag {row['Lag (months)']}: F={row['F-statistic']:.3f}, p={row['p-value']:.4f}"
                )

        lines.append("\n\n─" * 35)
        lines.append("FULL RESULTS TABLE")
        lines.append("─" * 70)
        lines.append(results_df.to_string(index=False))

        # Count
        total_tests = len(results_df)
        sig_count = len(sig)
        lines.append(f"\n\nTotal tests: {total_tests}")
        lines.append(
            f"Significant (p<0.05): {sig_count} ({sig_count / total_tests * 100:.1f}%)"
        )
        lines.append(
            f"Significant (p<0.01): {len(results_df[results_df['Significant (p<0.01)'] == 'Yes'])} tests"
        )

        # Bonferroni correction note
        bonferroni = 0.05 / total_tests if total_tests > 0 else 0.05
        lines.append(f"\nNote: With Bonferroni correction, α = {bonferroni:.5f}")
        sig_bonferroni = results_df[results_df["p-value"] < bonferroni]
        lines.append(f"Significant after Bonferroni: {len(sig_bonferroni)} tests")

    report_path = REPORTS / "granger_causality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    if not results_df.empty:
        results_df.to_csv(REPORTS / "granger_causality_results.csv", index=False)
        print(f"  Saved: {REPORTS / 'granger_causality_results.csv'}")


def main() -> None:
    print("=" * 70)
    print("GRANGER CAUSALITY PIPELINE")
    print("=" * 70)

    df = load_data()
    results_df = run_granger_all(df)

    visualize_granger(results_df)
    generate_report(results_df)

    print("\n" + "=" * 70)
    print("GRANGER CAUSALITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
