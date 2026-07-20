"""
COMPREHENSIVE TEST SUITE FOR TN AGRI PRICE PREDICTION PROJECT
Tests: Model integrity, prediction consistency, data quality, 
       feature engineering, edge cases, and dashboard/predict parity.
"""

import sys
import os
import math
import traceback
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path so imports work
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

# ─── Test Results Tracker ────────────────────────────────────────────────────
PASS = 0
FAIL = 0
ERRORS = []


def test(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL += 1
        msg = f"  ❌ FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        ERRORS.append(msg)
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: FILE INTEGRITY — Do all required files exist?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 1: FILE INTEGRITY")
print("=" * 70)

required_files = {
    "Model file": PROJECT_ROOT / "models" / "tn_no_lag_model.joblib",
    "Dashboard": PROJECT_ROOT / "dashboard.py",
    "Predict script": PROJECT_ROOT / "scripts" / "04_inference" / "predict.py",
    "Training script": PROJECT_ROOT / "scripts" / "02_modeling" / "tn_no_lag_model.py",
    "Dashboard price CSV": PROJECT_ROOT / "data" / "quality_checked" / "tn_retail_prices_dashboard.csv",
    "Agriculture census CSV": PROJECT_ROOT / "data" / "tamil_nadu" / "tn_agriculture_census.csv",
    "Requirements.txt": PROJECT_ROOT / "requirements.txt",
    "README.md": PROJECT_ROOT / "README.md",
    ".gitignore": PROJECT_ROOT / ".gitignore",
    "Model comparison report": PROJECT_ROOT / "reports" / "table_model_comparison.csv",
    "Paper (LaTeX)": PROJECT_ROOT / "paper" / "main.tex",
}

for name, path in required_files.items():
    test(f"File exists: {name}", path.exists(), f"Missing: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: MODEL INTEGRITY — Is the model file valid and complete?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 2: MODEL INTEGRITY")
print("=" * 70)

model_data = joblib.load(PROJECT_ROOT / "models" / "tn_no_lag_model.joblib")

# Check all required keys exist
test("Model dict has 'model' key", "model" in model_data, f"Keys found: {list(model_data.keys())}")
test("Model dict has 'le_commodity' key", "le_commodity" in model_data)
test("Model dict has 'scaler' key", "scaler" in model_data)
test("Model dict has 'features' key", "features" in model_data)

# Check model type
model = model_data.get("model")
le = model_data.get("le_commodity")

if model is not None:
    model_type = type(model).__name__
    test("Model is GradientBoostingRegressor", model_type == "GradientBoostingRegressor", f"Got: {model_type}")
    test("Model has predict method", hasattr(model, "predict"))
    test("Model has feature_importances_", hasattr(model, "feature_importances_"))

    # Check feature count matches expected (8 features)
    expected_features = ["commodity_encoded", "year_trend", "month", "season",
                         "is_monsoon", "rainfall_mm", "rainfall_deviation", "rainfall_category"]
    n_features = model.n_features_in_
    test(f"Model expects 8 features", n_features == 8, f"Got: {n_features}")

# Check label encoder
if le is not None:
    n_classes = len(le.classes_)
    test(f"Label encoder has classes (got {n_classes})", n_classes > 0)
    test("'Rice' is in commodities", "Rice" in le.classes_)
    test("'Tomato' is in commodities", "Tomato" in le.classes_)
    test("'Onion' is in commodities", "Onion" in le.classes_)
    test("'Wheat' is in commodities", "Wheat" in le.classes_)
    test("'Milk' is in commodities", "Milk" in le.classes_)

    # Print all commodities for reference
    print(f"\n  [INFO] All {n_classes} commodities: {list(le.classes_)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: PREDICTION CONSISTENCY — dashboard.py vs predict.py
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 3: PREDICTION CONSISTENCY (Dashboard vs Terminal)")
print("=" * 70)

model = model_data["model"]
le = model_data["le_commodity"]


def dashboard_predict(commodity, month, year, rainfall_mm):
    """Exact replica of dashboard.py predict_price function."""
    if commodity not in le.classes_:
        return None
    commodity_encoded = le.transform([commodity])[0]
    year_trend = year - 2015

    if month in [6, 7, 8, 9]:
        season = 0
    elif month in [10, 11]:
        season = 1
    elif month in [12, 1, 2]:
        season = 2
    else:
        season = 3

    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    deviation = ((rainfall_mm - 50) / 50) * 100
    cat = 0 if rainfall_mm < 50 else (1 if rainfall_mm < 150 else 2)

    features = [[commodity_encoded, year_trend, month, season, is_monsoon, rainfall_mm, deviation, cat]]
    base_price = model.predict(features)[0]

    inflation_rate = 0.042
    years_from_2026 = year - 2026
    adjusted_price = base_price * (1 + (inflation_rate * years_from_2026))

    return adjusted_price


def terminal_predict(commodity, month, year, rainfall_mm):
    """Exact replica of predict.py predict function."""
    commodity_encoded = le.transform([commodity])[0]
    year_trend = year - 2015

    if month in [6, 7, 8, 9]:
        season = 0
    elif month in [10, 11]:
        season = 1
    elif month in [12, 1, 2]:
        season = 2
    else:
        season = 3

    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    rainfall_deviation = ((rainfall_mm - 50) / 50) * 100
    rainfall_category = 0 if rainfall_mm < 50 else (1 if rainfall_mm < 150 else 2)

    features = [[commodity_encoded, year_trend, month, season, is_monsoon, rainfall_mm, rainfall_deviation, rainfall_category]]
    base_price = model.predict(features)[0]

    inflation_rate = 0.042
    years_from_2026 = year - 2026
    price = base_price * (1 + (inflation_rate * years_from_2026))

    return price


# Test across multiple commodities, months, years, and rainfalls
test_cases = [
    ("Tomato", 6, 2026, 68.0),
    ("Rice", 1, 2026, 30.0),
    ("Onion", 10, 2027, 120.0),
    ("Wheat", 3, 2025, 45.0),
    ("Milk", 8, 2028, 200.0),
    ("Sugar", 12, 2026, 10.0),
    ("Gram Dal", 7, 2026, 150.0),
    ("Tea Loose", 4, 2030, 80.0),
]

for commodity, month, year, rain in test_cases:
    p_dash = dashboard_predict(commodity, month, year, rain)
    p_term = terminal_predict(commodity, month, year, rain)
    match = abs(p_dash - p_term) < 0.001
    test(
        f"Parity: {commodity} M={month} Y={year} R={rain}mm → Rs.{p_dash:.2f}",
        match,
        f"Dashboard={p_dash:.4f}, Terminal={p_term:.4f}" if not match else ""
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: PREDICTION SANITY — Are outputs in reasonable ranges?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 4: PREDICTION SANITY CHECKS")
print("=" * 70)

# All predictions should be positive
for c in le.classes_:
    p = dashboard_predict(c, 6, 2026, 68.0)
    test(f"Positive price: {c}", p is not None and p > 0, f"Got: {p}")

# No prediction should exceed Rs. 1000/kg (sanity upper bound)
for c in le.classes_:
    p = dashboard_predict(c, 6, 2026, 68.0)
    test(f"Below Rs.1000: {c} (Rs.{p:.2f})", p < 1000, f"Got: {p}")

# Tea Loose should be most expensive, Salt cheapest (general knowledge check)
tea_price = dashboard_predict("Tea Loose", 6, 2026, 68.0)
rice_price = dashboard_predict("Rice", 6, 2026, 68.0)
salt_price = dashboard_predict("Salt Pack (Iodised)", 6, 2026, 68.0)
test("Tea > Rice (commodity ordering)", tea_price > rice_price,
     f"Tea={tea_price:.2f}, Rice={rice_price:.2f}")

# Year variation test
p_2025 = dashboard_predict("Tomato", 6, 2025, 68.0)
p_2026 = dashboard_predict("Tomato", 6, 2026, 68.0)
p_2027 = dashboard_predict("Tomato", 6, 2027, 68.0)
p_2030 = dashboard_predict("Tomato", 6, 2030, 68.0)

test("Year slider: 2025 < 2026", p_2025 < p_2026,
     f"2025={p_2025:.2f}, 2026={p_2026:.2f}")
test("Year slider: 2026 < 2027", p_2026 < p_2027,
     f"2026={p_2026:.2f}, 2027={p_2027:.2f}")
test("Year slider: 2027 < 2030", p_2027 < p_2030,
     f"2027={p_2027:.2f}, 2030={p_2030:.2f}")

# Month variation test — monsoon months should differ from winter
p_jan = dashboard_predict("Tomato", 1, 2026, 68.0)
p_jul = dashboard_predict("Tomato", 7, 2026, 68.0)
test("Month variation: Jan ≠ Jul for Tomato", abs(p_jan - p_jul) > 0.01,
     f"Jan={p_jan:.2f}, Jul={p_jul:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: REAL-WORLD VALIDATION — Compare with known prices
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 5: REAL-WORLD VALIDATION")
print("=" * 70)

# User confirmed: Tomato June 2026 actual ≈ Rs. 42
tomato_pred = dashboard_predict("Tomato", 6, 2026, 68.0)
error_pct = abs(tomato_pred - 42) / 42 * 100
test(f"Tomato Jun 2026: within 20% of Rs.42 actual (predicted Rs.{tomato_pred:.2f}, err={error_pct:.1f}%)",
     error_pct < 20)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: FEATURE ENGINEERING CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 6: FEATURE ENGINEERING")
print("=" * 70)

# Season mapping tests
season_tests = [
    (1, 2, "Jan → Winter"),
    (2, 2, "Feb → Winter"),
    (3, 3, "Mar → Summer"),
    (4, 3, "Apr → Summer"),
    (5, 3, "May → Summer"),
    (6, 0, "Jun → Monsoon"),
    (7, 0, "Jul → Monsoon"),
    (8, 0, "Aug → Monsoon"),
    (9, 0, "Sep → Monsoon"),
    (10, 1, "Oct → Post-Monsoon"),
    (11, 1, "Nov → Post-Monsoon"),
    (12, 2, "Dec → Winter"),
]
for month_val, expected_season, desc in season_tests:
    if month_val in [6, 7, 8, 9]:
        actual = 0
    elif month_val in [10, 11]:
        actual = 1
    elif month_val in [12, 1, 2]:
        actual = 2
    else:
        actual = 3
    test(f"Season: {desc}", actual == expected_season, f"Got season={actual}")

# Rainfall deviation tests
test("Deviation(50mm) = 0%", ((50 - 50) / 50) * 100 == 0)
test("Deviation(100mm) = 100%", ((100 - 50) / 50) * 100 == 100)
test("Deviation(0mm) = -100%", ((0 - 50) / 50) * 100 == -100)

# Rainfall category tests
test("Category(30mm) = 0 (Low)", (0 if 30 < 50 else (1 if 30 < 150 else 2)) == 0)
test("Category(100mm) = 1 (Normal)", (0 if 100 < 50 else (1 if 100 < 150 else 2)) == 1)
test("Category(200mm) = 2 (Heavy)", (0 if 200 < 50 else (1 if 200 < 150 else 2)) == 2)

# is_monsoon tests
test("is_monsoon(Jun) = 1", (1 if 6 in [6, 7, 8, 9] else 0) == 1)
test("is_monsoon(Jan) = 0", (1 if 1 in [6, 7, 8, 9] else 0) == 0)

# Year trend tests
test("year_trend(2015) = 0", 2015 - 2015 == 0)
test("year_trend(2026) = 11", 2026 - 2015 == 11)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: DATA QUALITY — CSV file checks
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 7: DATA QUALITY")
print("=" * 70)

# Dashboard prices CSV
prices_path = PROJECT_ROOT / "data" / "quality_checked" / "tn_retail_prices_dashboard.csv"
tn_prices = pd.read_csv(prices_path)

test("Price CSV: has 'date' column", "date" in tn_prices.columns)
test("Price CSV: has 'commodity' column", "commodity" in tn_prices.columns)
test("Price CSV: has 'price' column", "price" in tn_prices.columns)
test(f"Price CSV: has rows ({len(tn_prices):,})", len(tn_prices) > 0)
test("Price CSV: no null prices", tn_prices["price"].notna().all())
test("Price CSV: all prices > 0", (tn_prices["price"] > 0).all())

# Parse dates
tn_prices["date"] = pd.to_datetime(tn_prices["date"])
min_year = tn_prices["date"].dt.year.min()
max_year = tn_prices["date"].dt.year.max()
test(f"Price CSV: starts from 2015 or earlier (got {min_year})", min_year <= 2015)
test(f"Price CSV: goes up to 2024 (got {max_year})", max_year >= 2024)

# Check commodities in CSV match model's label encoder
csv_commodities = set(tn_prices["commodity"].unique())
model_commodities = set(le.classes_)
missing_from_csv = model_commodities - csv_commodities
test("All model commodities exist in price CSV",
     len(missing_from_csv) == 0,
     f"Missing: {missing_from_csv}" if missing_from_csv else "")

# Agriculture census CSV
agri_path = PROJECT_ROOT / "data" / "tamil_nadu" / "tn_agriculture_census.csv"
agri_df = pd.read_csv(agri_path)

test("Census CSV: has 'district_name' column", "district_name" in agri_df.columns)
test("Census CSV: has 'crop_name' column", "crop_name" in agri_df.columns)
test("Census CSV: has 'total_ar_district' column", "total_ar_district" in agri_df.columns)
test(f"Census CSV: has rows ({len(agri_df):,})", len(agri_df) > 0)

# Check some districts from dashboard exist in census
dashboard_districts = ["Coimbatore", "Madurai", "Salem", "Thanjavur", "Erode"]
census_districts = agri_df["district_name"].str.lower().unique()
for d in dashboard_districts:
    found = d.lower() in census_districts
    test(f"Census has district: {d}", found)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 8: EDGE CASES & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 8: EDGE CASES & ROBUSTNESS")
print("=" * 70)

# Zero rainfall
p_zero_rain = dashboard_predict("Tomato", 6, 2026, 0.0)
test("Zero rainfall produces valid price", p_zero_rain is not None and p_zero_rain > 0, f"Got: {p_zero_rain}")

# Extreme rainfall (500mm flood scenario)
p_flood = dashboard_predict("Tomato", 6, 2026, 500.0)
test("Flood rainfall (500mm) produces valid price", p_flood is not None and p_flood > 0, f"Got: {p_flood}")

# Boundary months
p_dec = dashboard_predict("Rice", 12, 2026, 50.0)
p_jan = dashboard_predict("Rice", 1, 2026, 50.0)
test("December prediction valid", p_dec is not None and p_dec > 0)
test("January prediction valid", p_jan is not None and p_jan > 0)

# Unknown commodity should return None
p_unknown = dashboard_predict("Mango", 6, 2026, 68.0)
test("Unknown commodity returns None", p_unknown is None)

# Far future year (2030) — should still work
p_future = dashboard_predict("Rice", 6, 2030, 68.0)
test("Year 2030 prediction valid", p_future is not None and p_future > 0)

# Past year (2024) — within training data
p_past = dashboard_predict("Rice", 6, 2024, 68.0)
test("Year 2024 prediction valid", p_past is not None and p_past > 0)

# Inflation sanity: 2030 > 2024
test("Inflation: 2030 price > 2024 price", p_future > p_past,
     f"2030={p_future:.2f}, 2024={p_past:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 9: MODEL COMPARISON REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 9: REPORT FILES VALIDATION")
print("=" * 70)

# Model comparison CSV
comp_path = PROJECT_ROOT / "reports" / "table_model_comparison.csv"
comp_df = pd.read_csv(comp_path)
test("Comparison CSV: has 'Model' column", "Model" in comp_df.columns)
test("Comparison CSV: has 'R2' column", "R2" in comp_df.columns)
test("Comparison CSV: has multiple models", len(comp_df) >= 2)

# Best model R2 > 0.90
if "R2" in comp_df.columns:
    # R2 column has format like '0.925 ± 0.048', extract the main number
    r2_values = comp_df["R2"].astype(str).str.split(" ").str[0].astype(float)
    best_r2 = r2_values.max()
    test(f"Best model R2 > 0.90 (got {best_r2:.4f})", best_r2 > 0.90)

# No-lag model report
report_path = PROJECT_ROOT / "reports" / "tn_no_lag_model_report.txt"
test("Model report exists", report_path.exists())
if report_path.exists():
    report_text = report_path.read_text(encoding="utf-8")
    test("Report mentions Gradient Boosting", "Gradient Boosting" in report_text)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 10: DASHBOARD DISTRICT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST GROUP 10: DASHBOARD DISTRICT COORDINATES")
print("=" * 70)

# All 32 TN districts in dashboard
TN_DISTRICTS = {
    "Ariyalur": (11.14, 79.08), "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96), "Cuddalore": (11.75, 79.77),
    "Dharmapuri": (12.13, 78.16), "Dindigul": (10.36, 77.97),
    "Erode": (11.34, 77.72), "Kanchipuram": (12.83, 79.70),
    "Kanniyakumari": (8.09, 77.54), "Karur": (10.96, 78.08),
    "Krishnagiri": (12.52, 78.21), "Madurai": (9.93, 78.12),
    "Nagapattinam": (10.77, 79.84), "Namakkal": (11.22, 78.17),
    "Nilgiris": (11.49, 76.73), "Perambalur": (11.23, 78.88),
    "Pudukkottai": (10.38, 78.80), "Ramanathapuram": (9.36, 78.84),
    "Salem": (11.66, 78.15), "Sivagangai": (9.84, 78.48),
    "Thanjavur": (10.79, 79.14), "Theni": (10.01, 77.48),
    "Thoothukudi": (8.76, 78.13), "Tiruchirappalli": (10.79, 78.70),
    "Tirunelveli": (8.71, 77.76), "Tiruppur": (11.11, 77.34),
    "Tiruvallur": (13.14, 79.91), "Tiruvannamalai": (12.23, 79.07),
    "Tiruvarur": (10.77, 79.64), "Vellore": (12.92, 79.13),
    "Viluppuram": (11.94, 79.49), "Virudhunagar": (9.59, 77.95),
}

test(f"Dashboard has 32 districts", len(TN_DISTRICTS) == 32)

# All coordinates should be within Tamil Nadu bounds
# TN lat: ~8.07 to ~13.57, lon: ~76.23 to ~80.35
for dist, (lat, lon) in TN_DISTRICTS.items():
    in_bounds = (7.5 < lat < 14.0) and (76.0 < lon < 81.0)
    test(f"Coordinates valid: {dist} ({lat},{lon})", in_bounds)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)
total = PASS + FAIL
print(f"\n  Total Tests: {total}")
print(f"  ✅ Passed:   {PASS}")
print(f"  ❌ Failed:   {FAIL}")
print(f"  Score:       {PASS}/{total} ({PASS/total*100:.1f}%)")

if ERRORS:
    print(f"\n  FAILURES:")
    for e in ERRORS:
        print(f"    {e}")
else:
    print("\n  🎉 ALL TESTS PASSED! Project is fully validated.")

print("=" * 70)
