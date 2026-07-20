"""
PER-COMMODITY ML PIPELINE (Publication-Quality)
=================================================
Trains separate models for each commodity to eliminate the commodity-dominance
problem. Forces models to learn genuine climate-price dynamics.

This is the PRIMARY contribution pipeline for the paper.

Outputs:
  - reports/per_commodity_results.csv
  - reports/per_commodity_detailed.txt
  - visualizations/figure_per_commodity_performance.png
  - visualizations/figure_per_commodity_shap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import sys
import io
import warnings
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
        "figure.autolayout": True,
    }
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_QUALITY = PROJECT_ROOT / "data" / "quality_checked"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS = PROJECT_ROOT / "visualizations"
REPORTS = PROJECT_ROOT / "reports"
INTERMEDIATE = PROJECT_ROOT / "data" / "intermediate"

for d in [MODELS_DIR, VISUALIZATIONS, REPORTS, INTERMEDIATE]:
    d.mkdir(parents=True, exist_ok=True)

# Key commodities for Tamil Nadu study
KEY_COMMODITIES = ["Rice", "Wheat", "Onion", "Tomato", "Potato", "Sugar", "Milk"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ── Metrics ────────────────────────────────────────────────────────────────────


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, safe against zero division."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, MAE, MAPE."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


# ── Data Loading ───────────────────────────────────────────────────────────────


def load_commodity_data() -> pd.DataFrame:
    """
    Load retail prices and rainfall for Tamil Nadu.
    Merge on (year, month) and engineer features.
    Returns one DataFrame with all commodities.
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # ── Prices ─────────────────────────────────────────────────────────────
    prices = pd.read_csv(DATA_QUALITY / "retail_prices_quality.csv")
    prices["date"] = pd.to_datetime(prices["date"])

    # Filter Tamil Nadu
    tn_prices = prices[prices["state_name"] == "Tamil Nadu"].copy()
    print(f"  TN price records: {len(tn_prices):,}")

    # Monthly aggregation per commodity
    tn_prices["year"] = tn_prices["date"].dt.year
    tn_prices["month"] = tn_prices["date"].dt.month
    price_monthly = (
        tn_prices.groupby(["commodity", "year", "month"])
        .agg(price=("price", "mean"))
        .reset_index()
    )

    # ── Rainfall ───────────────────────────────────────────────────────────
    try:
        rainfall = pd.read_csv(DATA_RAW / "daily-rainfall-data-district-level.csv")
        tn_rain = rainfall[
            rainfall["state_name"].str.contains("Tamil", case=False, na=False)
        ].copy()
        tn_rain["date"] = pd.to_datetime(tn_rain["date"])
        print(f"  TN district rainfall records: {len(tn_rain):,}")
    except (FileNotFoundError, KeyError):
        rainfall = pd.read_csv(DATA_QUALITY / "rainfall_state_quality.csv")
        tn_rain = rainfall[rainfall["state_name"] == "Tamil Nadu"].copy()
        tn_rain["date"] = pd.to_datetime(tn_rain["date"])
        print(f"  TN state rainfall records: {len(tn_rain):,}")

    tn_rain["year"] = tn_rain["date"].dt.year
    tn_rain["month"] = tn_rain["date"].dt.month

    if "actual" in tn_rain.columns:
        rain_monthly = (
            tn_rain.groupby(["year", "month"])
            .agg(
                rainfall_mm=("actual", "sum"),
                rainfall_deviation=("deviation", "mean"),
            )
            .reset_index()
        )
    else:
        rain_monthly = (
            tn_rain.groupby(["year", "month"])
            .agg(
                rainfall_mm=("rainfall", "sum"),
            )
            .reset_index()
        )
        rain_monthly["rainfall_deviation"] = 0.0

    # ── Merge ──────────────────────────────────────────────────────────────
    merged = price_monthly.merge(rain_monthly, on=["year", "month"], how="left")
    merged["rainfall_mm"] = merged["rainfall_mm"].fillna(merged["rainfall_mm"].median())
    merged["rainfall_deviation"] = merged["rainfall_deviation"].fillna(0)

    # ── Feature Engineering (NO commodity feature) ─────────────────────────
    merged["year_trend"] = merged["year"] - merged["year"].min()
    merged["season"] = merged["month"].apply(
        lambda m: (
            0
            if m in [6, 7, 8, 9]
            else (1 if m in [10, 11] else (2 if m in [12, 1, 2] else 3))
        )
    )
    merged["is_monsoon"] = merged["month"].isin([6, 7, 8, 9]).astype(int)

    # Cyclical encoding for month (better than raw integer for tree models)
    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

    # Rainfall categories
    merged["rainfall_category"] = (
        pd.cut(
            merged["rainfall_mm"], bins=[-1, 50, 150, 300, 100000], labels=[0, 1, 2, 3]
        )
        .astype(float)
        .fillna(1)
        .astype(int)
    )

    # Sort chronologically
    merged = merged.sort_values(["commodity", "year", "month"]).reset_index(drop=True)

    print(f"  Total merged records: {len(merged):,}")
    print(f"  Commodities available: {merged['commodity'].nunique()}")
    print(f"  Year range: {merged['year'].min()} – {merged['year'].max()}")

    return merged


# ── Per-Commodity Training ─────────────────────────────────────────────────────

FEATURES = [
    "year_trend",
    "month",
    "month_sin",
    "month_cos",
    "season",
    "is_monsoon",
    "rainfall_mm",
    "rainfall_deviation",
    "rainfall_category",
]


def get_models() -> Dict[str, object]:
    """Return a dictionary of models to benchmark."""
    return {
        "Naive Mean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }


def train_single_commodity(
    df_commodity: pd.DataFrame,
    commodity_name: str,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Optional[object], Optional[StandardScaler]]:
    """
    Train and evaluate all models for a single commodity using TimeSeriesSplit.

    Returns:
        fold_results: DataFrame with per-fold, per-model metrics.
        best_model: The best fitted model on all data.
        scaler: The fitted StandardScaler.
    """
    df = df_commodity.sort_values(["year", "month"]).reset_index(drop=True)

    if len(df) < 20:
        print(f"  [SKIP] {commodity_name}: only {len(df)} records (need ≥20)")
        return pd.DataFrame(), None, None

    X = df[FEATURES].values
    y = df["price"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = get_models()
    scaler = StandardScaler()

    fold_results = []
    all_predictions: Dict[str, List[float]] = {name: [] for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        for model_name, model in models.items():
            # Some models work better with scaled data
            needs_scaling = model_name in ("Ridge", "Naive Mean")
            X_tr = X_train_sc if needs_scaling else X_train
            X_te = X_test_sc if needs_scaling else X_test

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            metrics = compute_metrics(y_test, y_pred)
            fold_results.append(
                {
                    "Commodity": commodity_name,
                    "Model": model_name,
                    "Fold": fold_idx + 1,
                    **metrics,
                }
            )
            all_predictions[model_name].extend(y_pred.tolist())

    fold_df = pd.DataFrame(fold_results)

    # ── Statistical significance (Wilcoxon) ────────────────────────────────
    if not fold_df.empty:
        mean_r2 = fold_df.groupby("Model")["R2"].mean()
        best_model_name = mean_r2.idxmax()

        for model_name in models:
            if model_name != best_model_name and len(
                all_predictions[model_name]
            ) == len(all_predictions[best_model_name]):
                try:
                    _, p_val = stats.wilcoxon(
                        all_predictions[model_name],
                        all_predictions[best_model_name],
                    )
                    fold_df.loc[fold_df["Model"] == model_name, "p_value"] = p_val
                except ValueError:
                    fold_df.loc[fold_df["Model"] == model_name, "p_value"] = np.nan
            else:
                fold_df.loc[fold_df["Model"] == model_name, "p_value"] = np.nan
    else:
        best_model_name = "XGBoost"

    # ── Train final best model on all data ─────────────────────────────────
    scaler_final = StandardScaler()
    X_all_sc = scaler_final.fit_transform(X)

    final_model = get_models()[best_model_name]
    needs_scaling = best_model_name in ("Ridge", "Naive Mean")
    final_model.fit(X_all_sc if needs_scaling else X, y)

    return fold_df, final_model, scaler_final


def run_all_commodities(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Run per-commodity pipeline for all key commodities.

    Returns:
        all_results: Combined fold-level results DataFrame.
        commodity_models: Dict mapping commodity -> (model, scaler).
    """
    print("\n" + "=" * 70)
    print("PER-COMMODITY MODEL TRAINING (TimeSeriesSplit CV)")
    print("=" * 70)

    all_results = []
    commodity_models = {}

    available = df["commodity"].unique()
    commodities_to_run = [c for c in KEY_COMMODITIES if c in available]

    if not commodities_to_run:
        print("  [WARNING] None of the key commodities found. Running all available.")
        commodities_to_run = sorted(available)[:10]

    for commodity in commodities_to_run:
        print(f"\n  ── {commodity} {'─' * (50 - len(commodity))}")
        df_c = df[df["commodity"] == commodity]
        fold_df, model, scaler = train_single_commodity(df_c, commodity)

        if not fold_df.empty:
            all_results.append(fold_df)
            commodity_models[commodity] = (model, scaler)

            # Print summary
            summary = (
                fold_df.groupby("Model")
                .agg(
                    R2_mean=("R2", "mean"),
                    R2_std=("R2", "std"),
                    RMSE_mean=("RMSE", "mean"),
                    MAPE_mean=("MAPE", "mean"),
                )
                .sort_values("R2_mean", ascending=False)
            )

            best = summary.index[0]
            r2 = summary.loc[best, "R2_mean"]
            rmse = summary.loc[best, "RMSE_mean"]
            print(f"    Best: {best} | R²={r2:.3f} | RMSE={rmse:.2f}")

    combined = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )
    return combined, commodity_models


# ── SHAP Analysis ──────────────────────────────────────────────────────────────


def generate_shap_per_commodity(
    df: pd.DataFrame,
    commodity_models: Dict,
) -> None:
    """Generate SHAP summary plots in a grid for all commodities."""
    if not HAS_SHAP:
        print("\n  [SKIP] SHAP not installed. Run: pip install shap")
        return

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY (Per Commodity)")
    print("=" * 70)

    commodities = [c for c in KEY_COMMODITIES if c in commodity_models]
    n = len(commodities)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, commodity in enumerate(commodities):
        model, scaler = commodity_models[commodity]
        df_c = df[df["commodity"] == commodity].sort_values(["year", "month"])
        X = df_c[FEATURES].values

        # Use tree explainer for tree models, otherwise kernel
        try:
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X[:200])
            else:
                background = shap.sample(X, min(50, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X[:100])

            ax = axes[idx]
            shap.summary_plot(
                shap_values,
                X[: len(shap_values)],
                feature_names=FEATURES,
                show=False,
                plot_size=None,
                max_display=len(FEATURES),
            )
            # Move the SHAP plot to our subplot
            current_fig = plt.gcf()
            if current_fig != fig:
                plt.close(current_fig)
            ax.set_title(commodity, fontweight="bold", fontsize=11)

        except Exception as e:
            print(f"    [SHAP ERROR] {commodity}: {e}")
            axes[idx].text(
                0.5, 0.5, f"{commodity}\nSHAP failed", ha="center", va="center"
            )
            axes[idx].set_title(commodity, fontweight="bold")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "SHAP Feature Importance by Commodity", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        VISUALIZATIONS / "figure_per_commodity_shap.png", dpi=300, bbox_inches="tight"
    )
    plt.close("all")
    print(f"  Saved: {VISUALIZATIONS / 'figure_per_commodity_shap.png'}")


# ── Feature Importance (Tree-based) ────────────────────────────────────────────


def generate_feature_importance_table(
    df: pd.DataFrame,
    commodity_models: Dict,
) -> pd.DataFrame:
    """Build a table of feature importances per commodity (tree models only)."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE TABLE")
    print("=" * 70)

    rows = []
    for commodity, (model, _scaler) in commodity_models.items():
        if hasattr(model, "feature_importances_"):
            for feat, imp in zip(FEATURES, model.feature_importances_):
                rows.append(
                    {
                        "Commodity": commodity,
                        "Feature": feat,
                        "Importance": round(imp, 4),
                    }
                )

    if not rows:
        print("  No tree-based models with feature importances found.")
        return pd.DataFrame()

    fi_df = pd.DataFrame(rows)

    # Pivot for readability
    pivot = fi_df.pivot(
        index="Feature", columns="Commodity", values="Importance"
    ).fillna(0)
    pivot["Mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Mean", ascending=False)

    print(pivot.to_string())

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_plot = pivot.drop(columns="Mean")
    pivot_plot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        "Feature Importance by Commodity (Without Commodity Feature)", fontweight="bold"
    )
    ax.legend(title="Commodity", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        VISUALIZATIONS / "figure_feature_importance_per_commodity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_feature_importance_per_commodity.png'}")

    return fi_df


# ── Results Visualization ──────────────────────────────────────────────────────


def visualize_results(results_df: pd.DataFrame) -> None:
    """Create publication-quality performance comparison chart."""
    if results_df.empty:
        return

    print("\n" + "=" * 70)
    print("GENERATING PERFORMANCE VISUALIZATIONS")
    print("=" * 70)

    # Aggregate: mean ± std per (Commodity, Model)
    summary = (
        results_df.groupby(["Commodity", "Model"])
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
        )
        .reset_index()
    )

    # ── Figure: R² comparison across commodities ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric, label in zip(
        axes,
        ["R2_mean", "RMSE_mean", "MAPE_mean"],
        ["R² Score", "RMSE (₹)", "MAPE (%)"],
    ):
        pivot = summary.pivot(index="Commodity", columns="Model", values=metric)
        # Keep only the top 4 models for clarity
        top_models = summary.groupby("Model")[metric].mean()
        if metric == "R2_mean":
            top_models = top_models.nlargest(4)
        else:
            top_models = top_models.nsmallest(4)
        pivot = pivot[[c for c in top_models.index if c in pivot.columns]]

        pivot.plot(kind="bar", ax=ax, width=0.8, rot=45)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Commodity", fontweight="bold")
        ax.legend(title="Model", fontsize=8)

    plt.suptitle(
        "Per-Commodity Model Performance (5-Fold TimeSeriesSplit)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        VISUALIZATIONS / "figure_per_commodity_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved: {VISUALIZATIONS / 'figure_per_commodity_performance.png'}")


# ── Report Generation ──────────────────────────────────────────────────────────


def generate_report(results_df: pd.DataFrame, fi_df: pd.DataFrame) -> None:
    """Save per-commodity results for the paper."""
    if results_df.empty:
        print("  No results to save.")
        return

    # ── CSV summary ────────────────────────────────────────────────────────
    summary = (
        results_df.groupby(["Commodity", "Model"])
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
        )
        .round(4)
        .reset_index()
    )

    summary.to_csv(REPORTS / "per_commodity_results.csv", index=False)
    print(f"  Saved: {REPORTS / 'per_commodity_results.csv'}")

    # ── Best model per commodity ───────────────────────────────────────────
    best_per_commodity = summary.loc[summary.groupby("Commodity")["R2_mean"].idxmax()][
        ["Commodity", "Model", "R2_mean", "R2_std", "RMSE_mean", "MAPE_mean"]
    ]

    # ── Text report ────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("PER-COMMODITY MODEL PERFORMANCE (Publication Report)")
    lines.append("=" * 70)
    lines.append(f"\nEvaluation: 5-Fold TimeSeriesSplit | Seed: {RANDOM_SEED}")
    lines.append(f"Features: {', '.join(FEATURES)}")
    lines.append("NOTE: Commodity is NOT used as a feature.\n")

    lines.append("─" * 70)
    lines.append("BEST MODEL PER COMMODITY")
    lines.append("─" * 70)
    lines.append(best_per_commodity.to_string(index=False))

    lines.append("\n" + "─" * 70)
    lines.append("FULL RESULTS (ALL MODELS × ALL COMMODITIES)")
    lines.append("─" * 70)
    lines.append(summary.to_string(index=False))

    if not fi_df.empty:
        lines.append("\n" + "─" * 70)
        lines.append("MEAN FEATURE IMPORTANCE ACROSS COMMODITIES")
        lines.append("─" * 70)
        mean_fi = (
            fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
        )
        lines.append(mean_fi.to_string())

    report_path = REPORTS / "per_commodity_detailed.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    # ── Save models ────────────────────────────────────────────────────────
    # (done in main via commodity_models)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the complete per-commodity pipeline."""
    print("=" * 70)
    print("PER-COMMODITY ML PIPELINE (Publication Quality)")
    print("No commodity feature — forces climate/temporal learning")
    print("=" * 70)

    # Load
    df = load_commodity_data()

    # Train
    results_df, commodity_models = run_all_commodities(df)

    if results_df.empty:
        print("\n[ERROR] No results produced. Check data availability.")
        return

    # Feature importance
    fi_df = generate_feature_importance_table(df, commodity_models)

    # SHAP
    generate_shap_per_commodity(df, commodity_models)

    # Visualize
    visualize_results(results_df)

    # Report
    generate_report(results_df, fi_df)

    # Save all models
    model_save_path = MODELS_DIR / "per_commodity_models.joblib"
    joblib.dump(
        {
            "models": {k: (m, s) for k, (m, s) in commodity_models.items()},
            "features": FEATURES,
            "seed": RANDOM_SEED,
        },
        model_save_path,
    )
    print(f"  Saved models: {model_save_path}")

    print("\n" + "=" * 70)
    print("PER-COMMODITY PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
