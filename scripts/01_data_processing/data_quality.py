"""
Data Quality Processing Script
- Handle missing values
- Remove duplicates
- Standardize formats
- Validate data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import io

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_QUALITY = PROJECT_ROOT / "data" / "quality_checked"
REPORTS = PROJECT_ROOT / "reports"

# Create directories
DATA_QUALITY.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# ============================================================
# 1. MISSING VALUE HANDLERS
# ============================================================
def handle_missing_values(df, config):
    """
    Handle missing values based on column-specific strategies.

    config format: {'column_name': 'strategy'}
    strategies: 'drop', 'zero', 'mean', 'median', 'mode', 'ffill', 'bfill'
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    for col, strategy in config.items():
        if col not in df.columns:
            continue

        if strategy == "drop":
            df = df.dropna(subset=[col])
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if len(df[col].mode()) > 0 else np.nan
            )
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()

    missing_after = df.isnull().sum().sum()
    print(f"  Missing values: {missing_before:,} → {missing_after:,}")
    return df


# ============================================================
# 2. DUPLICATE REMOVAL
# ============================================================
def remove_duplicates(df, subset=None, keep="first"):
    """Remove duplicate rows."""
    rows_before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    rows_after = len(df)
    removed = rows_before - rows_after
    print(f"  Duplicates removed: {removed:,} ({removed/rows_before*100:.2f}%)")
    return df


# ============================================================
# 3. FORMAT STANDARDIZATION
# ============================================================
def standardize_dates(df, date_cols):
    """Standardize date columns to datetime format."""
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def standardize_state_names(df, col="state_name"):
    """Standardize state names to title case with consistent naming."""
    if col not in df.columns:
        return df

    df[col] = df[col].astype(str).str.strip().str.title()

    # Standard replacements for consistency
    replacements = {
        "Andaman And Nicobar Islands": "Andaman & Nicobar",
        "Andaman & Nicobar Islands": "Andaman & Nicobar",
        "Jammu And Kashmir": "Jammu & Kashmir",
        "Dadra And Nagar Haveli": "Dadra & Nagar Haveli",
        "Daman And Diu": "Daman & Diu",
        "Nct Of Delhi": "Delhi",
        "Delhi (Nct)": "Delhi",
        "Orissa": "Odisha",
        "Pondicherry": "Puducherry",
        "Uttaranchal": "Uttarakhand",
    }
    df[col] = df[col].replace(replacements)
    return df


def standardize_numeric(df, numeric_cols):
    """Ensure numeric columns are proper numeric type."""
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_columns(df):
    """Standardize column names."""
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


# ============================================================
# 4. DATA VALIDATION
# ============================================================
def validate_data(df, rules):
    """
    Validate data against rules and return report.

    rules format: {'column': {'min': val, 'max': val, 'not_null': True, 'unique': True}}
    """
    issues = []

    for col, checks in rules.items():
        if col not in df.columns:
            issues.append(f"Column '{col}' not found in dataset")
            continue

        if checks.get("not_null"):
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"'{col}': {null_count:,} null values found")

        if "min" in checks:
            violations = (df[col] < checks["min"]).sum()
            if violations > 0:
                issues.append(
                    f"'{col}': {violations:,} values below min ({checks['min']})"
                )

        if "max" in checks:
            violations = (df[col] > checks["max"]).sum()
            if violations > 0:
                issues.append(
                    f"'{col}': {violations:,} values above max ({checks['max']})"
                )

        if checks.get("unique"):
            duplicates = df[col].duplicated().sum()
            if duplicates > 0:
                issues.append(f"'{col}': {duplicates:,} duplicate values")

    return issues


def generate_quality_report(datasets_info):
    """Generate a comprehensive data quality report."""
    report = []
    report.append("=" * 70)
    report.append("DATA QUALITY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    for name, info in datasets_info.items():
        report.append(f"\n[DATASET] {name}")
        report.append("-" * 50)
        report.append(f"  Rows: {info['rows_before']:,} -> {info['rows_after']:,}")
        report.append(
            f"  Missing values: {info['missing_before']:,} -> {info['missing_after']:,}"
        )
        report.append(f"  Duplicates removed: {info['duplicates_removed']:,}")

        if info["validation_issues"]:
            report.append("  [WARNING] Validation issues:")
            for issue in info["validation_issues"]:
                report.append(f"    - {issue}")
        else:
            report.append("  [OK] All validations passed")

    return "\n".join(report)


# ============================================================
# MAIN PROCESSING
# ============================================================
def process_dataset(
    filepath,
    name,
    missing_config,
    date_cols,
    numeric_cols,
    validation_rules,
    dedup_cols=None,
):
    """Process a single dataset through all quality steps."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print("=" * 60)

    # Load
    df = pd.read_csv(filepath)
    rows_before = len(df)
    missing_before = df.isnull().sum().sum()

    # 1. Standardize columns
    print("1. Standardizing columns...")
    df = standardize_columns(df)

    # 2. Standardize dates
    print("2. Standardizing dates...")
    df = standardize_dates(df, date_cols)

    # 3. Standardize state names
    print("3. Standardizing state names...")
    df = standardize_state_names(df)

    # 4. Standardize numeric
    print("4. Standardizing numeric columns...")
    df = standardize_numeric(df, numeric_cols)

    # 5. Handle missing values
    print("5. Handling missing values...")
    df = handle_missing_values(df, missing_config)

    # 6. Remove duplicates
    print("6. Removing duplicates...")
    duplicates_before = len(df)
    df = remove_duplicates(df, subset=dedup_cols)
    duplicates_removed = duplicates_before - len(df)

    # 7. Validate
    print("7. Validating data...")
    issues = validate_data(df, validation_rules)
    for issue in issues:
        print(f"  [WARNING] {issue}")

    rows_after = len(df)
    missing_after = df.isnull().sum().sum()

    return df, {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_removed": duplicates_removed,
        "validation_issues": issues,
    }


def main():
    """Process all datasets."""
    print("\n" + "=" * 70)
    print("STARTING DATA QUALITY PROCESSING")
    print("=" * 70)

    datasets_info = {}

    # ===== 1. MSP =====
    df, info = process_dataset(
        DATA_PROCESSED / "msp_cleaned.csv",
        "Minimum Support Prices",
        missing_config={"min_support_price": "ffill"},
        date_cols=[],
        numeric_cols=["min_support_price", "year_start"],
        validation_rules={
            "min_support_price": {"min": 0, "not_null": True},
            "crop": {"not_null": True},
            "season": {"not_null": True},
        },
        dedup_cols=["year", "crop", "season"],
    )
    df.to_csv(DATA_QUALITY / "msp_quality.csv", index=False)
    datasets_info["MSP"] = info

    # ===== 2. Cost of Cultivation =====
    df, info = process_dataset(
        DATA_PROCESSED / "cost_of_cultivation_cleaned.csv",
        "Cost of Cultivation",
        missing_config={
            "cul_cost_c2": "median",
            "main_product_value": "median",
            "derived_yield": "median",
            "profit_margin_pct": "zero",
        },
        date_cols=[],
        numeric_cols=[
            "cul_cost_c2",
            "main_product_value",
            "derived_yield",
            "profit_margin_pct",
        ],
        validation_rules={
            "state_name": {"not_null": True},
            "crop_name": {"not_null": True},
            "cul_cost_c2": {"min": 0},
        },
        dedup_cols=["year", "state_name", "crop_name"],
    )
    df.to_csv(DATA_QUALITY / "cost_of_cultivation_quality.csv", index=False)
    datasets_info["Cost of Cultivation"] = info

    # ===== 3. Rainfall =====
    df, info = process_dataset(
        DATA_PROCESSED / "rainfall_state_cleaned.csv",
        "State Rainfall",
        missing_config={"actual": "zero", "normal": "median", "deviation": "zero"},
        date_cols=["date"],
        numeric_cols=["actual", "normal", "deviation"],
        validation_rules={
            "state_name": {"not_null": True},
            "date": {"not_null": True},
            "actual": {"min": 0},
        },
        dedup_cols=["date", "state_name"],
    )
    df.to_csv(DATA_QUALITY / "rainfall_state_quality.csv", index=False)
    datasets_info["State Rainfall"] = info

    # ===== 4. Groundwater =====
    df, info = process_dataset(
        DATA_PROCESSED / "groundwater_cleaned.csv",
        "Groundwater Levels",
        missing_config={"currentlevel": "median", "level_diff": "zero"},
        date_cols=["date"],
        numeric_cols=["currentlevel", "level_diff", "latitude", "longitude"],
        validation_rules={
            "state_name": {"not_null": True},
            "currentlevel": {"min": 0, "max": 500},
        },
        dedup_cols=["date", "state_name", "district_name", "station_name"],
    )
    df.to_csv(DATA_QUALITY / "groundwater_quality.csv", index=False)
    datasets_info["Groundwater"] = info

    # ===== 5. Vulnerability =====
    df, info = process_dataset(
        DATA_PROCESSED / "vulnerability_state_cleaned.csv",
        "Vulnerability Indicators",
        missing_config={
            "climate_vul_in": "median",
            "vulnerability_normalized": "median",
        },
        date_cols=[],
        numeric_cols=["climate_vul_in", "vulnerability_normalized"],
        validation_rules={
            "state_name": {"not_null": True, "unique": True},
            "climate_vul_in": {"min": 0, "max": 1},
        },
        dedup_cols=["state_name"],
    )
    df.to_csv(DATA_QUALITY / "vulnerability_quality.csv", index=False)
    datasets_info["Vulnerability"] = info

    # ===== 6. Retail Prices (sample for speed) =====
    print(f"\n{'='*60}")
    print("Processing: Retail Prices (large file - may take time)")
    print("=" * 60)

    df = pd.read_csv(DATA_PROCESSED / "retail_prices_cleaned.csv")
    rows_before = len(df)
    missing_before = df.isnull().sum().sum()

    df = standardize_columns(df)
    df = standardize_dates(df, ["date"])
    df = standardize_state_names(df)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Fill missing prices with commodity-state median
    df["price"] = df.groupby(["commodity", "state_name"])["price"].transform(
        lambda x: x.fillna(x.median())
    )
    # Remaining nulls fill with commodity median
    df["price"] = df.groupby("commodity")["price"].transform(
        lambda x: x.fillna(x.median())
    )

    duplicates_before = len(df)
    df = df.drop_duplicates(subset=["date", "state_name", "commodity"], keep="first")
    duplicates_removed = duplicates_before - len(df)

    # Remove invalid prices
    df = df[(df["price"] > 0) & (df["price"] < 10000)]

    df.to_csv(DATA_QUALITY / "retail_prices_quality.csv", index=False)

    datasets_info["Retail Prices"] = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "missing_before": missing_before,
        "missing_after": df.isnull().sum().sum(),
        "duplicates_removed": duplicates_removed,
        "validation_issues": [],
    }
    print(f"  Rows: {rows_before:,} → {len(df):,}")
    print(f"  Missing: {missing_before:,} → {df.isnull().sum().sum():,}")
    print(f"  Duplicates removed: {duplicates_removed:,}")

    # Generate report
    report = generate_quality_report(datasets_info)
    print("\n" + report)

    # Save report
    report_path = REPORTS / "data_quality_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 70)
    print("DATA QUALITY PROCESSING COMPLETE!")
    print(f"Quality-checked files saved to: {DATA_QUALITY}")
    print("=" * 70)


if __name__ == "__main__":
    main()
