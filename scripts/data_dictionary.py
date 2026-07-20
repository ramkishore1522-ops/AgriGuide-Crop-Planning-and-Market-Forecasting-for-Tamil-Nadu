"""
Data Dictionary for Agricultural Analysis Project
Contains metadata for all datasets from India Data Portal
"""

from datetime import date

DATA_DICTIONARY = {
    # ============== PRICE DATA ==============
    "daily-retail-prices-of-essential-commodities.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 2117473,
        "columns": ["id", "date", "state_name", "state_code", "commodity", "price"],
        "date_range": "2015-01-01 onwards",
        "spatial_granularity": "State-level",
        "temporal_granularity": "Daily",
        "missing_data": {"price": 79821},
        "notes": "Essential commodity retail prices across states",
        "integration_keys": ["state_name", "state_code", "date", "commodity"]
    },
    
    "wholesale-prices-in-india.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "file_size_mb": 4.6,
        "spatial_granularity": "National/Market-level",
        "notes": "Wholesale price indices and commodity prices"
    },
    
    "wholesale-prices-state-level.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "file_size_mb": 122,
        "spatial_granularity": "State-level",
        "notes": "State-wise wholesale price data"
    },
    
    "minimum-support-prices.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 737,
        "columns": ["id", "year", "crop", "season", "min_support_price"],
        "date_range": "Historical MSP data up to 2022-2023",
        "spatial_granularity": "National",
        "temporal_granularity": "Yearly by season",
        "missing_data": {"min_support_price": 1},
        "notes": "Government minimum support prices for Kharif/Rabi crops",
        "integration_keys": ["year", "crop", "season"]
    },
    
    "cost-of-cultivation.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 3880,
        "columns": [
            "year", "state_name", "crop_name", "crop_type",
            "cul_cost_a1", "cul_cost_a2", "cul_cost_b1", "cul_cost_b2",
            "cul_cost_c1", "cul_cost_c2", "cul_cost_c2rev",
            "prod_cost_a1", "prod_cost_a2", "prod_cost_b1", "prod_cost_b2",
            "prod_cost_c1", "prod_cost_c2", "prod_cost_c2rev", "prod_cost_c3",
            "main_product_value", "by_product_value",
            "opr_cost", "fix_cost", "derived_yield"
        ],
        "date_range": "2018-19 onwards",
        "spatial_granularity": "State-level",
        "temporal_granularity": "Yearly",
        "missing_data": {"opr_cost_crop_insurance": 2915, "opr_cost_contractor_pay": 3743},
        "notes": "Detailed cost breakdown for major crops (A1, A2, B1, B2, C1, C2, C3 costs)",
        "integration_keys": ["year", "state_name", "crop_name"]
    },
    
    "consumer-price-index (1).csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 291890,
        "columns": ["id", "date", "state_name", "state_code", "commodity_group", "sector", "cpi", "inflation_rate"],
        "date_range": "2014-01-01 onwards",
        "spatial_granularity": "State-level + All India",
        "temporal_granularity": "Monthly",
        "missing_data": {"cpi": 2822, "inflation_rate": 2822},
        "notes": "CPI by commodity group (Cereals, Clothing, etc.) and sector (Rural/Urban/Combined)",
        "integration_keys": ["date", "state_name", "commodity_group", "sector"]
    },
    
    # ============== CLIMATE & WATER DATA ==============
    "daily-rainfall-at-state-level.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 204876,
        "columns": ["id", "date", "state_code", "state_name", "actual", "rfs", "normal", "deviation"],
        "date_range": "2009-01-01 onwards",
        "spatial_granularity": "State-level",
        "temporal_granularity": "Daily",
        "missing_data": {"actual": 17162, "rfs": 5865, "normal": 11518, "deviation": 31021},
        "notes": "Daily rainfall with actual vs normal comparison and deviation percentage",
        "integration_keys": ["date", "state_name", "state_code"]
    },
    
    "daily-rainfall-data-district-level.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 3577399,
        "columns": ["id", "date", "state_code", "state_name", "district_code", "district_name", 
                   "actual", "rfs", "normal", "deviation"],
        "date_range": "2009-04-01 onwards",
        "spatial_granularity": "District-level",
        "temporal_granularity": "Daily",
        "missing_data": {"actual": 139887, "rfs": 41812, "normal": 665655, "deviation": 1145632},
        "notes": "Most granular rainfall data - district-wise daily readings",
        "integration_keys": ["date", "state_name", "district_name", "district_code"]
    },
    
    "cgwb-changes-in-depth-to-water-level.csv": {
        "source": "India Data Portal (CGWB)",
        "date_downloaded": "2026-01-21",
        "records": 550850,
        "columns": ["id", "date", "state_name", "state_code", "district_name", "district_code",
                   "station_name", "latitude", "longitude", "basin", "sub_basin", 
                   "source", "currentlevel", "level_diff"],
        "date_range": "2013-11-04 onwards",
        "spatial_granularity": "Station/District-level with coordinates",
        "temporal_granularity": "Bi-annual (Pre/Post monsoon)",
        "missing_data": {},
        "notes": "Groundwater depth measurements from CGWB monitoring stations. level_diff shows change from previous period.",
        "integration_keys": ["date", "state_name", "district_name", "district_code"]
    },
    
    "climate-vulnerability-indicators-district-wise.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 699,
        "num_columns": 79,
        "key_indicators": [
            "climate_vul_in",  # Climate Vulnerability Index
            "yield_variability",
            "area_rainfed_agri",
            "groundwater_extracted",
            "groundwater_available",
            "population_multi_hazard",
            "water_scarcity",
            "soil_fertility",
            "crop_divsi_in"
        ],
        "date_range": "2021 (single snapshot)",
        "spatial_granularity": "District-level",
        "missing_data": "High - many districts have incomplete indicator data",
        "notes": "Comprehensive vulnerability assessment with 79 indicators. Many NaN values for specific indicators.",
        "integration_keys": ["state_name", "district_name", "district_code"]
    },
    
    "climate-vulnerability-indicators-state-wise.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 36,
        "num_columns": 29,
        "key_indicators": [
            "climate_vul_in",
            "yield_variability",
            "rainfed_agriculture",
            "irrigated_area",
            "poverty_rate",
            "women_workforce"
        ],
        "date_range": "2019 (single snapshot)",
        "spatial_granularity": "State-level",
        "missing_data": {"all_indicators": 6},  # 6 states/UTs missing data
        "notes": "State-level vulnerability summary - useful for cross-state comparisons",
        "integration_keys": ["state_name", "state_code"]
    },
    
    # ============== AGRICULTURE PRODUCTION DATA ==============
    "district-level-agcensus-crop.csv": {
        "source": "India Data Portal (Ag Census)",
        "date_downloaded": "2026-01-21",
        "records": 1689936,
        "columns": ["year", "state_name", "district_name", "social_group", "farm_size_class",
                   "farm_size_category", "crop_name", "crop_code", "crop_type",
                   "hold_no_district", "irr_ar_district", "unirr_ar_district", "total_ar_district"],
        "date_range": "2010-11",
        "spatial_granularity": "District-level",
        "temporal_granularity": "Census period (decadal)",
        "notes": "Detailed crop area by farm size, social group. irr_ar = irrigated area, unirr_ar = unirrigated",
        "integration_keys": ["year", "state_name", "district_name", "crop_name"]
    },
    
    "state-level-agcensus-crop.csv": {
        "source": "India Data Portal (Ag Census)",
        "date_downloaded": "2026-01-21",
        "file_size_mb": 23,
        "spatial_granularity": "State-level",
        "notes": "State-level aggregate of crop census data"
    },
    
    "tehsil-level-agcensus-crop.csv": {
        "source": "India Data Portal (Ag Census)",
        "date_downloaded": "2026-01-21",
        "file_size_mb": 905,
        "spatial_granularity": "Tehsil-level",
        "notes": "Most granular crop census - very large file, process in chunks"
    },
    
    "area-and-production-statistics-nhb.csv": {
        "source": "India Data Portal (National Horticulture Board)",
        "date_downloaded": "2026-01-21",
        "records": 1270,
        "columns": ["id", "year", "state_name", "state_code", "crop_category", "area", "production"],
        "date_range": "2018-19 onwards",
        "spatial_granularity": "State-level",
        "temporal_granularity": "Yearly",
        "notes": "Horticulture production - Fruits and Vegetables area and production",
        "integration_keys": ["year", "state_name", "crop_category"]
    },
    
    # ============== EXPORT DATA ==============
    "exports-to-asian-countries.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 5348874,
        "columns": ["date", "country_name", "alpha_3_code", "region", "sub_region",
                   "hs_code", "commodity", "unit", "value_qt", "value_rs", "value_dl"],
        "date_range": "2015-01-01 onwards",
        "spatial_granularity": "Country-level",
        "temporal_granularity": "Monthly",
        "missing_data": {"unit": 5838, "value_qt": 45289, "value_dl": 282888},
        "notes": "Largest export dataset. value_qt = quantity, value_rs = INR, value_dl = USD",
        "integration_keys": ["date", "country_name", "hs_code", "commodity"]
    },
    
    "exports-to-european-countries.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 2827471,
        "columns": ["date", "country_name", "hs_code", "commodity", "value_qt", "value_rs", "value_dl"],
        "date_range": "2015-01-01 onwards",
        "notes": "European exports including UK, Germany, France, Italy etc."
    },
    
    "exports-to-african-countries.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 2715238,
        "columns": ["date", "country_name", "hs_code", "commodity", "value_qt", "value_rs", "value_dl"],
        "date_range": "2015-01-01 onwards",
        "notes": "African exports - major markets include Nigeria, South Africa, Egypt"
    },
    
    "exports-to-american-countries.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 1763960,
        "columns": ["date", "country_name", "hs_code", "commodity", "value_qt", "value_rs", "value_dl"],
        "date_range": "2015-01-01 onwards",
        "notes": "American exports - includes USA, Canada, Brazil, Mexico"
    },
    
    "exports-to-oceanic-countries.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 463608,
        "columns": ["date", "country_name", "hs_code", "commodity", "value_qt", "value_rs", "value_dl"],
        "date_range": "2015-01-01 onwards",
        "notes": "Oceanic exports - primarily Australia and New Zealand"
    },
    
    # ============== OTHER DATA ==============
    "monthly-food-distribution-data.csv": {
        "source": "India Data Portal",
        "date_downloaded": "2026-01-21",
        "records": 63425,
        "columns": ["month", "state_name", "district_name", 
                   "total_rice_allocated", "total_wheat_allocated",
                   "total_qty_distributed", "percent_qty_distributed"],
        "date_range": "2017-01-01 onwards",
        "spatial_granularity": "District-level",
        "temporal_granularity": "Monthly",
        "notes": "PDS food distribution - rice/wheat allocation and distribution"
    }
}

# Common crop mappings for integration
CROP_MAPPINGS = {
    "Paddy": ["Rice", "Paddy", "Paddy - Common", "Paddy - Grade 'A'"],
    "Wheat": ["Wheat"],
    "Maize": ["Maize", "Corn"],
    "Pulses": ["Tur (Arhar)", "Moong", "Urad", "Gram", "Masur"],
    "Oilseeds": ["Groundnut", "Soybean", "Sunflower", "Mustard", "Rapeseed"],
    "Commercial": ["Cotton", "Sugarcane", "Jute"]
}

# State code mappings
STATE_CODES = {
    1: "Jammu and Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Daman and Diu",
    26: "Dadra and Nagar Haveli",
    27: "Maharashtra",
    28: "Andhra Pradesh",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry",
    35: "Andaman and Nicobar Islands",
    36: "Telangana",
    37: "Ladakh"
}


def get_dataset_info(filename: str) -> dict:
    """Get metadata for a specific dataset."""
    return DATA_DICTIONARY.get(filename, {"error": f"Dataset {filename} not found"})


def get_integration_keys(filename: str) -> list:
    """Get the keys that can be used to merge this dataset with others."""
    info = DATA_DICTIONARY.get(filename, {})
    return info.get("integration_keys", [])


def print_dataset_summary():
    """Print a summary of all datasets."""
    print("=" * 80)
    print("AGRICULTURAL ANALYSIS PROJECT - DATASET SUMMARY")
    print("=" * 80)
    
    categories = {
        "Price Data": ["daily-retail-prices", "wholesale-prices", "minimum-support", "cost-of-cultivation", "consumer-price"],
        "Climate & Water": ["rainfall", "cgwb", "climate-vulnerability"],
        "Agriculture Production": ["agcensus", "area-and-production"],
        "Exports": ["exports-to"],
        "Other": ["monthly-food"]
    }
    
    for category, patterns in categories.items():
        print(f"\n📊 {category}")
        print("-" * 40)
        for filename, info in DATA_DICTIONARY.items():
            if any(p in filename for p in patterns):
                records = info.get("records", info.get("file_size_mb", "?"))
                print(f"  • {filename}: {records} records/MB")


if __name__ == "__main__":
    print_dataset_summary()
