"""
TAMIL NADU DISTRICT-LEVEL ML PIPELINE
Focused analysis for Tamil Nadu districts only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import io

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
DATA_TN = PROJECT_ROOT / 'data' / 'tamil_nadu'
MODELS_DIR = PROJECT_ROOT / 'models'
VISUALIZATIONS = PROJECT_ROOT / 'visualizations'
REPORTS = PROJECT_ROOT / 'reports'

# Create directories
DATA_TN.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Tamil Nadu Districts
TN_DISTRICTS = [
    'Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore', 
    'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kanchipuram',
    'Kanniyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai',
    'Nagapattinam', 'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai',
    'Ramanathapuram', 'Ranipet', 'Salem', 'Sivagangai', 'Tenkasi',
    'Thanjavur', 'Theni', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli',
    'Tirupathur', 'Tiruppur', 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur',
    'Vellore', 'Viluppuram', 'Virudhunagar'
]

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================
# 1. LOAD TAMIL NADU DATA
# ============================================================
def load_tamil_nadu_data():
    """Load and filter data for Tamil Nadu only."""
    print("\n" + "="*70)
    print("STEP 1: LOADING TAMIL NADU DATA")
    print("="*70)
    
    # Load retail prices for TN
    print("\n  Loading Retail Prices (Tamil Nadu)...")
    prices = pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv')
    tn_prices = prices[prices['state_name'] == 'Tamil Nadu'].copy()
    print(f"    Records: {len(tn_prices):,}")
    print(f"    Commodities: {tn_prices['commodity'].nunique()}")
    
    # Load district-level agriculture data
    print("\n  Loading District Agriculture Census...")
    agri = pd.read_csv(DATA_RAW / 'district-level-agcensus-crop.csv')
    tn_agri = agri[agri['state_name'] == 'Tamil Nadu'].copy()
    print(f"    Records: {len(tn_agri):,}")
    print(f"    Districts: {tn_agri['district_name'].nunique()}")
    print(f"    Crops: {tn_agri['crop_name'].nunique()}")
    
    # Load district rainfall
    print("\n  Loading District Rainfall...")
    try:
        rainfall = pd.read_csv(DATA_RAW / 'daily-rainfall-data-district-level.csv')
        tn_rainfall = rainfall[rainfall['state_name'].str.contains('Tamil', case=False, na=False)].copy()
        print(f"    Records: {len(tn_rainfall):,}")
    except:
        tn_rainfall = None
        print("    Not found - using state-level rainfall")
        rainfall_state = pd.read_csv(DATA_QUALITY / 'rainfall_state_quality.csv')
        tn_rainfall = rainfall_state[rainfall_state['state_name'] == 'Tamil Nadu'].copy()
        print(f"    State-level records: {len(tn_rainfall):,}")
    
    # Load vulnerability
    print("\n  Loading Vulnerability Indicators...")
    try:
        vuln = pd.read_csv(DATA_RAW / 'climate-vulnerability-indicators-district-wise.csv')
        tn_vuln = vuln[vuln['state_name'].str.contains('Tamil', case=False, na=False)].copy()
        print(f"    Districts with vulnerability data: {len(tn_vuln)}")
    except:
        tn_vuln = None
        print("    Using state-level vulnerability")
    
    # Save TN-specific data
    tn_prices.to_csv(DATA_TN / 'tn_retail_prices.csv', index=False)
    tn_agri.to_csv(DATA_TN / 'tn_agriculture_census.csv', index=False)
    print(f"\n  Saved Tamil Nadu data to: {DATA_TN}")
    
    return tn_prices, tn_agri, tn_rainfall, tn_vuln


# ============================================================
# 2. VISUALIZE TAMIL NADU DATA
# ============================================================
def visualize_tn_data(tn_prices, tn_agri):
    """Visualize Tamil Nadu specific data."""
    print("\n" + "="*70)
    print("STEP 2: VISUALIZING TAMIL NADU DATA")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tamil Nadu Agricultural Data Analysis', fontsize=14, fontweight='bold')
    
    # 1. Price distribution by commodity
    ax1 = axes[0, 0]
    top_commodities = tn_prices['commodity'].value_counts().head(10).index
    tn_top = tn_prices[tn_prices['commodity'].isin(top_commodities)]
    tn_top.boxplot(column='price', by='commodity', ax=ax1)
    ax1.set_xlabel('Commodity')
    ax1.set_ylabel('Price (Rs/kg)')
    ax1.set_title('Price Distribution by Commodity')
    plt.suptitle('')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. District crop area
    ax2 = axes[0, 1]
    district_area = tn_agri.groupby('district_name')['total_ar_district'].sum().sort_values(ascending=False).head(15)
    district_area.plot(kind='barh', ax=ax2, color='green', alpha=0.7)
    ax2.set_xlabel('Total Crop Area')
    ax2.set_title('Top 15 Districts by Crop Area')
    ax2.invert_yaxis()
    
    # 3. Crop distribution
    ax3 = axes[1, 0]
    crop_area = tn_agri.groupby('crop_name')['total_ar_district'].sum().sort_values(ascending=False).head(10)
    crop_area.plot(kind='bar', ax=ax3, color='orange', alpha=0.7)
    ax3.set_xlabel('Crop')
    ax3.set_ylabel('Total Area')
    ax3.set_title('Top 10 Crops by Area in Tamil Nadu')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Price trends over time
    ax4 = axes[1, 1]
    tn_prices['date'] = pd.to_datetime(tn_prices['date'])
    tn_prices['year_month'] = tn_prices['date'].dt.to_period('M')
    price_trend = tn_prices.groupby('year_month')['price'].mean()
    price_trend.plot(ax=ax4, linewidth=2, color='blue')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Average Price (Rs/kg)')
    ax4.set_title('Price Trend Over Time')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'tn_data_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {VISUALIZATIONS / 'tn_data_analysis.png'}")


# ============================================================
# 3. PREPARE DATA FOR ML
# ============================================================
def prepare_tn_data(tn_prices):
    """Prepare Tamil Nadu data for ML models."""
    print("\n" + "="*70)
    print("STEP 3: PREPARING DATA FOR ML")
    print("="*70)
    
    df = tn_prices.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Aggregate
    df_agg = df.groupby(['commodity', 'year', 'month']).agg({
        'price': ['mean', 'std', 'min', 'max']
    }).reset_index()
    df_agg.columns = ['commodity', 'year', 'month', 'price_mean', 'price_std', 'price_min', 'price_max']
    df_agg['price_std'] = df_agg['price_std'].fillna(0)
    
    # Lag features
    df_agg = df_agg.sort_values(['commodity', 'year', 'month'])
    for lag in [1, 2, 3]:
        df_agg[f'price_lag{lag}'] = df_agg.groupby('commodity')['price_mean'].shift(lag)
    
    df_agg = df_agg.dropna()
    
    # Encode
    le_commodity = LabelEncoder()
    df_agg['commodity_encoded'] = le_commodity.fit_transform(df_agg['commodity'])
    
    # Features
    features = ['commodity_encoded', 'year', 'month', 'price_lag1', 'price_lag2', 'price_lag3']
    X = df_agg[features]
    y = df_agg['price_mean']
    
    print(f"\n  Prepared Dataset:")
    print(f"    Samples: {len(X):,}")
    print(f"    Features: {features}")
    print(f"    Commodities: {df_agg['commodity'].nunique()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"    Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le_commodity, features, df_agg


# ============================================================
# 4. TRAIN MODELS
# ============================================================
def train_tn_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train and compare models for Tamil Nadu."""
    print("\n" + "="*70)
    print("STEP 4: TRAINING MODELS FOR TAMIL NADU")
    print("="*70)
    
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=1.0), True),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), False)
    }
    
    results = []
    best_model = None
    best_r2 = -999
    
    print("\n  Training models...")
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        })
        
        print(f"    {name:20s}: R2={r2:.4f}, MAE={mae:.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = (name, model)
    
    print(f"\n  BEST MODEL: {best_model[0]} (R2={best_r2:.4f})")
    
    return pd.DataFrame(results), best_model


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================
def tune_tn_model(X_train, y_train):
    """Hyperparameter tuning for Tamil Nadu model."""
    print("\n" + "="*70)
    print("STEP 5: HYPERPARAMETER TUNING")
    print("="*70)
    
    # Grid Search
    print("\n  5.1 GRID SEARCH")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Best R2: {grid_search.best_score_:.4f}")
    
    # Randomized Search
    print("\n  5.2 RANDOMIZED SEARCH")
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(rf, param_dist, n_iter=15, cv=3, scoring='r2', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best R2: {random_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, random_search.best_estimator_


# ============================================================
# 6. DISTRICT-LEVEL ANALYSIS
# ============================================================
def analyze_districts(tn_agri):
    """Analyze and rank Tamil Nadu districts."""
    print("\n" + "="*70)
    print("STEP 6: DISTRICT-LEVEL ANALYSIS")
    print("="*70)
    
    # District summary
    district_summary = tn_agri.groupby('district_name').agg({
        'total_ar_district': 'sum',
        'irr_ar_district': 'sum',
        'unirr_ar_district': 'sum',
        'crop_name': 'nunique'
    }).reset_index()
    
    district_summary.columns = ['District', 'Total_Area', 'Irrigated_Area', 'Unirrigated_Area', 'Crop_Diversity']
    district_summary['Irrigation_Pct'] = (district_summary['Irrigated_Area'] / district_summary['Total_Area'] * 100).round(1)
    district_summary = district_summary.sort_values('Total_Area', ascending=False)
    
    print("\n  TOP 15 DISTRICTS BY AGRICULTURAL AREA:")
    print(district_summary.head(15).to_string(index=False))
    
    # Top crops per district
    print("\n  TOP CROPS BY DISTRICT:")
    top_crops = tn_agri.groupby(['district_name', 'crop_name'])['total_ar_district'].sum().reset_index()
    top_crops = top_crops.sort_values(['district_name', 'total_ar_district'], ascending=[True, False])
    
    district_crops = top_crops.groupby('district_name').head(3).groupby('district_name')['crop_name'].apply(list).reset_index()
    district_crops['top_crops'] = district_crops['crop_name'].apply(lambda x: ', '.join(x[:3]))
    
    for _, row in district_crops.head(15).iterrows():
        print(f"    {row['district_name']:20s}: {row['top_crops']}")
    
    # Save recommendations
    recommendations = district_summary.merge(district_crops[['district_name', 'top_crops']], left_on='District', right_on='district_name', how='left')
    recommendations = recommendations.drop(columns=['district_name'])
    recommendations.to_csv(REPORTS / 'tn_district_analysis.csv', index=False)
    print(f"\n  Saved: {REPORTS / 'tn_district_analysis.csv'}")
    
    return district_summary, recommendations


# ============================================================
# 7. GENERATE REPORT
# ============================================================
def generate_tn_report(model_results, district_summary, tn_prices, tn_agri):
    """Generate Tamil Nadu analysis report."""
    report = []
    report.append("="*70)
    report.append("TAMIL NADU DISTRICT-LEVEL AGRICULTURAL ANALYSIS")
    report.append("="*70)
    
    report.append("\n\n## DATA SUMMARY\n")
    report.append(f"- Retail Price Records: {len(tn_prices):,}")
    report.append(f"- Agriculture Census Records: {len(tn_agri):,}")
    report.append(f"- Districts Covered: {tn_agri['district_name'].nunique()}")
    report.append(f"- Commodities Tracked: {tn_prices['commodity'].nunique()}")
    report.append(f"- Crops in Census: {tn_agri['crop_name'].nunique()}")
    
    report.append("\n\n## MODEL PERFORMANCE\n")
    report.append(model_results.to_string(index=False))
    
    report.append("\n\n## TOP AGRICULTURAL DISTRICTS\n")
    report.append(district_summary.head(10).to_string(index=False))
    
    report_text = "\n".join(report)
    
    report_path = REPORTS / 'tn_ml_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n  Report saved: {report_path}")
    return report_text


# ============================================================
# MAIN
# ============================================================
def main():
    """Run Tamil Nadu ML pipeline."""
    print("\n" + "="*70)
    print("TAMIL NADU DISTRICT-LEVEL ML PIPELINE")
    print("="*70)
    
    # Load data
    tn_prices, tn_agri, tn_rainfall, tn_vuln = load_tamil_nadu_data()
    
    # Visualize
    visualize_tn_data(tn_prices, tn_agri)
    
    # Prepare
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le_commodity, features, df_agg = prepare_tn_data(tn_prices)
    
    # Train models
    model_results, best_model = train_tn_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Tune
    grid_model, random_model = tune_tn_model(X_train, y_train)
    
    # District analysis
    district_summary, recommendations = analyze_districts(tn_agri)
    
    # Generate report
    generate_tn_report(model_results, district_summary, tn_prices, tn_agri)
    
    # Save model
    model_path = MODELS_DIR / 'tn_price_model.joblib'
    joblib.dump({
        'model': grid_model,
        'scaler': scaler,
        'le_commodity': le_commodity,
        'features': features
    }, model_path)
    print(f"\n  Model saved: {model_path}")
    
    print("\n" + "="*70)
    print("TAMIL NADU ML PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
