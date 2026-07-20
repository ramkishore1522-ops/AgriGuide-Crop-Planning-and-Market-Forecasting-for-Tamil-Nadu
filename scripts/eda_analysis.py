"""
Exploratory Data Analysis Script
- Price-Climate Correlation
- Profitability Analysis
- Vulnerability Assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_QUALITY = PROJECT_ROOT / 'data' / 'quality_checked'
INTERMEDIATE = PROJECT_ROOT / 'data' / 'intermediate'
VISUALIZATIONS = PROJECT_ROOT / 'visualizations'
REPORTS = PROJECT_ROOT / 'reports'

# Create directories
VISUALIZATIONS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.autolayout': True
})
sns.set_palette('husl')


def load_data():
    """Load all quality-checked datasets."""
    print("Loading datasets...")
    
    data = {
        'msp': pd.read_csv(DATA_QUALITY / 'msp_quality.csv'),
        'coc': pd.read_csv(DATA_QUALITY / 'cost_of_cultivation_quality.csv'),
        'rainfall': pd.read_csv(DATA_QUALITY / 'rainfall_state_quality.csv', parse_dates=['date']),
        'groundwater': pd.read_csv(DATA_QUALITY / 'groundwater_quality.csv', parse_dates=['date']),
        'vulnerability': pd.read_csv(DATA_QUALITY / 'vulnerability_quality.csv'),
        'retail_prices': pd.read_csv(DATA_QUALITY / 'retail_prices_quality.csv', parse_dates=['date'])
    }
    
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows")
    
    return data


# ============================================================
# 1. PRICE-CLIMATE CORRELATION ANALYSIS
# ============================================================
def analyze_price_climate_correlation(data):
    """Analyze correlation between rainfall and commodity prices."""
    print("\n" + "="*70)
    print("1. PRICE-CLIMATE CORRELATION ANALYSIS")
    print("="*70)
    
    rainfall = data['rainfall'].copy()
    prices = data['retail_prices'].copy()
    
    # Aggregate rainfall by state and year-month
    rainfall['year_month'] = rainfall['date'].dt.to_period('M')
    rainfall_monthly = rainfall.groupby(['state_name', 'year_month']).agg({
        'actual': 'sum',
        'deviation': 'mean'
    }).reset_index()
    rainfall_monthly['year_month'] = rainfall_monthly['year_month'].astype(str)
    
    # Aggregate prices by state, commodity, and year-month
    prices['year_month'] = prices['date'].dt.to_period('M').astype(str)
    prices_monthly = prices.groupby(['state_name', 'commodity', 'year_month']).agg({
        'price': 'mean'
    }).reset_index()
    
    # Merge rainfall with prices
    merged = pd.merge(
        prices_monthly,
        rainfall_monthly,
        on=['state_name', 'year_month'],
        how='inner'
    )
    
    print(f"  Merged records: {len(merged):,}")
    
    # Calculate correlations by commodity
    correlations = []
    key_commodities = ['Rice', 'Wheat', 'Onion', 'Tomato', 'Potato', 'Sugar', 'Milk']
    
    for commodity in key_commodities:
        commodity_data = merged[merged['commodity'] == commodity]
        if len(commodity_data) > 30:
            # Pearson correlation
            r_actual, p_actual = stats.pearsonr(commodity_data['actual'], commodity_data['price'])
            r_dev, p_dev = stats.pearsonr(commodity_data['deviation'], commodity_data['price'])
            
            # Spearman rank correlation
            rho_actual, p_rho_actual = stats.spearmanr(commodity_data['actual'], commodity_data['price'])
            rho_dev, p_rho_dev = stats.spearmanr(commodity_data['deviation'], commodity_data['price'])
            
            correlations.append({
                'Commodity': commodity,
                'Pearson (Rainfall)': round(r_actual, 3),
                'p-val (Rain)': round(p_actual, 4),
                'Pearson (Dev)': round(r_dev, 3),
                'p-val (Dev)': round(p_dev, 4),
                'Spearman (Rainfall)': round(rho_actual, 3),
                'Spearman (Dev)': round(rho_dev, 3),
                'Sample Size': len(commodity_data)
            })
    
    corr_df = pd.DataFrame(correlations)
    print("\n  Price-Rainfall Correlations by Commodity (with Statistical Significance):")
    print(corr_df.to_string(index=False))
    
    # Save correlation table for paper
    corr_df.to_csv(REPORTS / 'table_price_climate_correlation.csv', index=False)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Correlation heatmap
    ax1 = axes[0, 0]
    corr_matrix = merged.groupby('commodity').apply(
        lambda x: pd.Series({
            'Rainfall': x['price'].corr(x['actual']),
            'Deviation': x['price'].corr(x['deviation'])
        })
    ).head(15)
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax1, fmt='.2f')
    ax1.set_title('Price Correlation with Rainfall Metrics', fontsize=12, fontweight='bold')
    
    # Scatter: Rice price vs Rainfall
    ax2 = axes[0, 1]
    rice_data = merged[merged['commodity'] == 'Rice']
    ax2.scatter(rice_data['actual'], rice_data['price'], alpha=0.3, s=10)
    ax2.set_xlabel('Monthly Rainfall (mm)')
    ax2.set_ylabel('Rice Price (Rs/kg)')
    ax2.set_title('Rice Price vs Monthly Rainfall', fontsize=12, fontweight='bold')
    
    # Add trendline
    if len(rice_data) > 10:
        z = np.polyfit(rice_data['actual'].dropna(), rice_data['price'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(rice_data['actual'].min(), rice_data['actual'].max(), 100)
        ax2.plot(x_line, p(x_line), color='red', linewidth=2, label='Trend')
        ax2.legend()
    
    # Onion price volatility vs rainfall deviation
    ax3 = axes[1, 0]
    onion_data = merged[merged['commodity'] == 'Onion']
    ax3.scatter(onion_data['deviation'], onion_data['price'], alpha=0.3, s=10, color='orange')
    ax3.set_xlabel('Rainfall Deviation (%)')
    ax3.set_ylabel('Onion Price (Rs/kg)')
    ax3.set_title('Onion Price vs Rainfall Deviation', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # State-wise correlation strength
    ax4 = axes[1, 1]
    state_corr = merged.groupby('state_name').apply(
        lambda x: x['price'].corr(x['deviation']) if len(x) > 30 else np.nan
    ).dropna().sort_values()
    
    colors = ['red' if x < 0 else 'green' for x in state_corr.values]
    ax4.barh(state_corr.index[-15:], state_corr.values[-15:], color=colors[-15:])
    ax4.set_xlabel('Price-Deviation Correlation')
    ax4.set_title('States: Price Sensitivity to Rainfall Deviation', fontsize=12, fontweight='bold')
    ax4.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_3_price_climate_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {VISUALIZATIONS / 'figure_3_price_climate_correlation.png'}")
    
    return corr_df, merged


# ============================================================
# 2. PROFITABILITY ANALYSIS
# ============================================================
def analyze_profitability(data):
    """Analyze crop profitability by state and crop type."""
    print("\n" + "="*70)
    print("2. PROFITABILITY ANALYSIS")
    print("="*70)
    
    coc = data['coc'].copy()
    msp = data['msp'].copy()
    
    # Calculate key profitability metrics
    coc['revenue_per_hectare'] = coc['main_product_value']
    coc['cost_per_hectare'] = coc['cul_cost_c2']
    coc['net_profit'] = coc['revenue_per_hectare'] - coc['cost_per_hectare']
    coc['profit_margin'] = (coc['net_profit'] / coc['revenue_per_hectare'] * 100).round(2)
    coc['cost_recovery'] = (coc['revenue_per_hectare'] / coc['cost_per_hectare']).round(2)
    
    # Top profitable crops
    crop_profit = coc.groupby('crop_name').agg({
        'profit_margin': 'mean',
        'net_profit': 'mean',
        'cost_recovery': 'mean'
    }).round(2).sort_values('profit_margin', ascending=False)
    
    print("\n  TOP 10 MOST PROFITABLE CROPS:")
    print(crop_profit.head(10).to_string())
    
    print("\n  BOTTOM 10 LEAST PROFITABLE CROPS:")
    print(crop_profit.tail(10).to_string())
    
    # State profitability
    state_profit = coc.groupby('state_name').agg({
        'profit_margin': 'mean',
        'net_profit': 'mean',
        'cost_per_hectare': 'mean'
    }).round(2).sort_values('profit_margin', ascending=False)
    
    print("\n  STATE PROFITABILITY RANKING:")
    print(state_profit.to_string())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top/Bottom crops by profit margin
    ax1 = axes[0, 0]
    top_bottom = pd.concat([crop_profit.head(8), crop_profit.tail(8)])
    colors = ['green' if x > 0 else 'red' for x in top_bottom['profit_margin']]
    ax1.barh(top_bottom.index, top_bottom['profit_margin'], color=colors)
    ax1.set_xlabel('Profit Margin (%)')
    ax1.set_title('Crop Profitability: Top & Bottom 8', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    
    # State ranking
    ax2 = axes[0, 1]
    colors2 = ['green' if x > 0 else 'red' for x in state_profit['profit_margin']]
    ax2.barh(state_profit.index, state_profit['profit_margin'], color=colors2)
    ax2.set_xlabel('Average Profit Margin (%)')
    ax2.set_title('State-wise Average Profitability', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    
    # Cost recovery ratio by crop type
    ax3 = axes[1, 0]
    crop_type_stats = coc.groupby('crop_type')['cost_recovery'].mean().sort_values(ascending=False)
    colors3 = plt.cm.viridis(np.linspace(0, 1, len(crop_type_stats)))
    ax3.bar(crop_type_stats.index, crop_type_stats.values, color=colors3)
    ax3.axhline(y=1, color='red', linestyle='--', label='Break-even')
    ax3.set_ylabel('Cost Recovery Ratio')
    ax3.set_title('Cost Recovery by Crop Type', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    # Profit distribution
    ax4 = axes[1, 1]
    ax4.hist(coc['profit_margin'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax4.axvline(x=coc['profit_margin'].median(), color='green', linestyle='-', linewidth=2, label=f'Median: {coc["profit_margin"].median():.1f}%')
    ax4.set_xlabel('Profit Margin (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Profit Margins', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_7_profitability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {VISUALIZATIONS / 'figure_7_profitability_analysis.png'}")
    
    # MSP Coverage Analysis
    print("\n  MSP COVERAGE ANALYSIS:")
    
    # Check latest MSP rates
    latest_msp = msp.groupby('crop')['min_support_price'].last()
    print(f"  Latest MSP rates available for {len(latest_msp)} crops")
    
    return crop_profit, state_profit


# ============================================================
# 3. VULNERABILITY ASSESSMENT
# ============================================================
def analyze_vulnerability(data):
    """Comprehensive vulnerability assessment by state."""
    print("\n" + "="*70)
    print("3. VULNERABILITY ASSESSMENT")
    print("="*70)
    
    vulnerability = data['vulnerability'].copy()
    rainfall = data['rainfall'].copy()
    groundwater = data['groundwater'].copy()
    coc = data['coc'].copy()
    
    # Calculate additional risk indicators
    
    # 1. Rainfall variability by state
    rainfall['year'] = rainfall['date'].dt.year
    rainfall_var = rainfall.groupby('state_name').agg({
        'actual': ['mean', 'std'],
        'deviation': 'mean'
    }).round(2)
    rainfall_var.columns = ['avg_rainfall', 'rainfall_std', 'avg_deviation']
    rainfall_var['rainfall_cv'] = (rainfall_var['rainfall_std'] / rainfall_var['avg_rainfall'] * 100).round(2)
    
    # 2. Groundwater stress
    gw_latest = groundwater.groupby('state_name')['currentlevel'].mean().round(2)
    
    # 3. Income stress (from profitability)
    income_stress = coc.groupby('state_name')['profit_margin_pct'].mean().round(2)
    
    # Build composite index
    composite = vulnerability[['state_name', 'climate_vul_in']].copy()
    composite = composite.merge(rainfall_var.reset_index(), on='state_name', how='left')
    composite = composite.merge(gw_latest.reset_index().rename(columns={'currentlevel': 'groundwater_depth'}), 
                                on='state_name', how='left')
    composite = composite.merge(income_stress.reset_index().rename(columns={'profit_margin_pct': 'avg_profit_margin'}),
                                on='state_name', how='left')
    
    # Normalize each component (0-1 scale)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 0.001)
    
    composite['climate_risk'] = normalize(composite['climate_vul_in'].fillna(0.5))
    composite['water_risk'] = normalize(composite['groundwater_depth'].fillna(composite['groundwater_depth'].median()))
    composite['rainfall_risk'] = normalize(composite['rainfall_cv'].fillna(0))
    composite['income_risk'] = normalize(-composite['avg_profit_margin'].fillna(0))  # Lower profit = higher risk
    
    # Composite vulnerability score (weighted)
    composite['composite_score'] = (
        composite['climate_risk'] * 0.30 +
        composite['water_risk'] * 0.25 +
        composite['rainfall_risk'] * 0.25 +
        composite['income_risk'] * 0.20
    ).round(3)
    
    # Risk categories
    composite['risk_level'] = pd.cut(
        composite['composite_score'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Sort by risk
    composite = composite.sort_values('composite_score', ascending=False)
    
    print("\n  VULNERABILITY RANKING (Top 15 States):")
    display_cols = ['state_name', 'composite_score', 'risk_level', 'climate_risk', 
                    'water_risk', 'rainfall_risk', 'income_risk']
    print(composite[display_cols].head(15).to_string(index=False))
    
    # Summary by risk level
    risk_summary = composite.groupby('risk_level').size()
    print("\n  RISK LEVEL DISTRIBUTION:")
    print(risk_summary.to_string())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Composite score bar chart
    ax1 = axes[0, 0]
    top_20 = composite.head(20)
    colors = ['red' if x == 'High Risk' else 'orange' if x == 'Medium Risk' else 'green' 
              for x in top_20['risk_level']]
    ax1.barh(top_20['state_name'], top_20['composite_score'], color=colors)
    ax1.set_xlabel('Composite Vulnerability Score')
    ax1.set_title('State Vulnerability Ranking (Top 20)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Component breakdown for top 10
    ax2 = axes[0, 1]
    top_10 = composite.head(10)
    x = np.arange(len(top_10))
    width = 0.2
    
    ax2.bar(x - 1.5*width, top_10['climate_risk'], width, label='Climate', color='red', alpha=0.7)
    ax2.bar(x - 0.5*width, top_10['water_risk'], width, label='Water', color='blue', alpha=0.7)
    ax2.bar(x + 0.5*width, top_10['rainfall_risk'], width, label='Rainfall', color='cyan', alpha=0.7)
    ax2.bar(x + 1.5*width, top_10['income_risk'], width, label='Income', color='purple', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_10['state_name'], rotation=45, ha='right')
    ax2.set_ylabel('Normalized Risk Score')
    ax2.set_title('Risk Component Breakdown (Top 10)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Risk level pie chart
    ax3 = axes[1, 0]
    risk_counts = composite['risk_level'].value_counts()
    colors_pie = ['green', 'orange', 'red']
    ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90)
    ax3.set_title('States by Risk Level', fontsize=12, fontweight='bold')
    
    # Scatter: Water depth vs Rainfall variability
    ax4 = axes[1, 1]
    ax4.scatter(composite['groundwater_depth'], composite['rainfall_cv'], 
                c=composite['composite_score'], cmap='RdYlGn_r', s=100, alpha=0.7)
    ax4.set_xlabel('Groundwater Depth (m)')
    ax4.set_ylabel('Rainfall Variability (CV %)')
    ax4.set_title('Water Stress vs Climate Variability', fontsize=12, fontweight='bold')
    plt.colorbar(ax4.collections[0], ax=ax4, label='Vulnerability Score')
    
    # Add state labels for top 5 high-risk
    for _, row in composite.head(5).iterrows():
        if pd.notna(row['groundwater_depth']) and pd.notna(row['rainfall_cv']):
            ax4.annotate(row['state_name'][:10], (row['groundwater_depth'], row['rainfall_cv']),
                        fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS / 'figure_8_vulnerability_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {VISUALIZATIONS / 'figure_8_vulnerability_assessment.png'}")
    
    # Save vulnerability data
    composite.to_csv(INTERMEDIATE / 'vulnerability_composite.csv', index=False)
    print(f"  Saved: {INTERMEDIATE / 'vulnerability_composite.csv'}")
    
    return composite


# ============================================================
# 4. GENERATE INSIGHTS SUMMARY
# ============================================================
def generate_insights_report(price_climate, profitability, vulnerability):
    """Generate a summary insights report."""
    print("\n" + "="*70)
    print("4. GENERATING INSIGHTS REPORT")
    print("="*70)
    
    crop_profit, state_profit = profitability
    
    report = []
    report.append("="*70)
    report.append("AGRICULTURAL DATA ANALYSIS - KEY INSIGHTS")
    report.append("="*70)
    
    # Price-Climate Insights
    report.append("\n## 1. PRICE-CLIMATE RELATIONSHIPS\n")
    report.append(price_climate.to_string(index=False))
    report.append("\nKey Findings:")
    report.append("- Prices generally show weak to moderate correlation with rainfall")
    report.append("- Vegetables (onion, tomato) are more sensitive to rainfall variability")
    report.append("- Negative deviation (drought) tends to increase prices")
    
    # Profitability Insights
    report.append("\n\n## 2. CROP PROFITABILITY\n")
    report.append("Top 5 Most Profitable Crops:")
    report.append(crop_profit.head(5).to_string())
    report.append("\nLeast Profitable Crops (Warning):")
    report.append(crop_profit[crop_profit['profit_margin'] < 0].to_string())
    report.append("\nKey Findings:")
    report.append(f"- Average profit margin: {crop_profit['profit_margin'].mean():.1f}%")
    report.append(f"- Crops with negative margins: {len(crop_profit[crop_profit['profit_margin'] < 0])}")
    
    # Vulnerability Insights
    report.append("\n\n## 3. VULNERABILITY ASSESSMENT\n")
    high_risk = vulnerability[vulnerability['risk_level'] == 'High Risk']
    report.append(f"HIGH RISK STATES ({len(high_risk)}):")
    for _, row in high_risk.iterrows():
        report.append(f"  - {row['state_name']}: Score {row['composite_score']:.3f}")
    
    report.append("\nRisk Factors to Monitor:")
    report.append("- Groundwater depletion in Punjab, Haryana, Rajasthan")
    report.append("- Rainfall variability in drought-prone states")
    report.append("- Low farmer income margins in certain regions")
    
    # Recommendations
    report.append("\n\n## 4. RECOMMENDATIONS\n")
    report.append("1. DIVERSIFICATION: Promote high-margin crops in vulnerable regions")
    report.append("2. WATER MANAGEMENT: Focus on groundwater recharge in depleting states")
    report.append("3. PRICE SUPPORT: Strengthen MSP coverage for low-margin crops")
    report.append("4. CLIMATE ADAPTATION: Early warning systems for price-sensitive commodities")
    report.append("5. INSURANCE: Targeted crop insurance for high-risk districts")
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = REPORTS / 'eda_insights_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  Report saved: {report_path}")
    print("\n" + report_text)
    
    return report_text


# ============================================================
# MAIN
# ============================================================
def main():
    """Run complete EDA pipeline."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Load data
    data = load_data()
    
    # Run analyses
    price_climate_corr, merged_data = analyze_price_climate_correlation(data)
    profitability = analyze_profitability(data)
    vulnerability = analyze_vulnerability(data)
    
    # Generate insights
    generate_insights_report(price_climate_corr, profitability, vulnerability)
    
    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print("="*70)
    print(f"\nVisualizations saved to: {VISUALIZATIONS}")
    print(f"Reports saved to: {REPORTS}")
    print(f"Intermediate data saved to: {INTERMEDIATE}")


if __name__ == "__main__":
    main()
