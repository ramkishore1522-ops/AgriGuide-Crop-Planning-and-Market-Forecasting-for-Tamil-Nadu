"""
INTERACTIVE PRICE PREDICTOR WITH LIVE WEATHER
Run this script and enter your own inputs!
"""

import sys
import os
from pathlib import Path
import joblib
import requests
import io

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

sys.path.append(str(PROJECT_ROOT / "scripts" / "04_inference"))
import live_macro_fetcher

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load model
model_data = joblib.load(PROJECT_ROOT / "models" / "tn_no_lag_model.joblib")
model_gb = model_data.get("model_gb")
model_ridge = model_data.get("model_ridge")
scaler = model_data.get("scaler")
le_commodity = model_data.get("le_commodity")

# All 32 Tamil Nadu Districts
TN_CITIES = {
    1: ("Ariyalur", 11.1400, 79.0800),
    2: ("Chennai", 13.0827, 80.2707),
    3: ("Coimbatore", 11.0168, 76.9558),
    4: ("Cuddalore", 11.7480, 79.7714),
    5: ("Dharmapuri", 12.1270, 78.1580),
    6: ("Dindigul", 10.3624, 77.9695),
    7: ("Erode", 11.3410, 77.7172),
    8: ("Kanchipuram", 12.8342, 79.7036),
    9: ("Kanniyakumari", 8.0883, 77.5385),
    10: ("Karur", 10.9601, 78.0766),
    11: ("Krishnagiri", 12.5186, 78.2137),
    12: ("Madurai", 9.9252, 78.1198),
    13: ("Nagapattinam", 10.7672, 79.8449),
    14: ("Namakkal", 11.2189, 78.1674),
    15: ("Nilgiris", 11.4916, 76.7337),
    16: ("Perambalur", 11.2340, 78.8800),
    17: ("Pudukkottai", 10.3833, 78.8001),
    18: ("Ramanathapuram", 9.3639, 78.8395),
    19: ("Salem", 11.6643, 78.1460),
    20: ("Sivagangai", 9.8433, 78.4809),
    21: ("Thanjavur", 10.7870, 79.1378),
    22: ("Theni", 10.0104, 77.4768),
    23: ("Thoothukudi", 8.7642, 78.1348),
    24: ("Tiruchirappalli", 10.7905, 78.7047),
    25: ("Tirunelveli", 8.7139, 77.7567),
    26: ("Tiruppur", 11.1085, 77.3411),
    27: ("Tiruvallur", 13.1439, 79.9086),
    28: ("Tiruvannamalai", 12.2253, 79.0747),
    29: ("Tiruvarur", 10.7713, 79.6370),
    30: ("Vellore", 12.9165, 79.1325),
    31: ("Viluppuram", 11.9401, 79.4861),
    32: ("Virudhunagar", 9.5851, 77.9526),
}


def get_live_weather(lat, lon):
    """Fetch live weather from Open-Meteo API."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "daily" in data:
            daily = data["daily"]
            weekly_rain = sum([p or 0 for p in daily["precipitation_sum"]])
            monthly_rain = weekly_rain * 4

            print("\n  LIVE WEATHER:")
            print("  " + "-" * 40)
            for i in range(min(5, len(daily["time"]))):
                date = daily["time"][i]
                rain = daily["precipitation_sum"][i] or 0
                tmax = daily["temperature_2m_max"][i]
                tmin = daily["temperature_2m_min"][i]
                print(f"    {date}: {tmin}-{tmax}C, Rain: {rain}mm")
            print("  " + "-" * 40)
            print(f"    Weekly rain: {weekly_rain:.1f}mm")
            print(f"    Monthly estimate: {monthly_rain:.1f}mm")

            return monthly_rain
    except Exception as e:
        print(f"  Weather API error: {e}")

    return None


def show_menu():
    print("\n" + "=" * 50)
    print("TAMIL NADU CROP PRICE PREDICTOR")
    print("=" * 50)
    print("\nAvailable Commodities:")
    for i, c in enumerate(le_commodity.classes_, 1):
        print(f"  {i:2}. {c}")
    print()


def show_cities():
    print("\nAvailable Cities:")
    for num, (name, lat, lon) in TN_CITIES.items():
        print(f"  {num}. {name}")
    print()


def get_user_input():
    # Get commodity
    print("Enter commodity number (1-22): ", end="")
    try:
        choice = int(input())
        if 1 <= choice <= len(le_commodity.classes_):
            commodity = le_commodity.classes_[choice - 1]
        else:
            print("Invalid choice!")
            return None
    except:
        print("Invalid input!")
        return None

    # Get city
    show_cities()
    print("Enter district number (1-32): ", end="")
    try:
        city_choice = int(input())
        if city_choice in TN_CITIES:
            city_name, lat, lon = TN_CITIES[city_choice]
        else:
            print("Invalid! Using Chennai.")
            city_name, lat, lon = TN_CITIES[2]
    except:
        print("Invalid! Using Chennai.")
        city_name, lat, lon = TN_CITIES[2]

    # Get month
    print("Enter month (1-12): ", end="")
    try:
        month = int(input())
        if not 1 <= month <= 12:
            print("Invalid month!")
            return None
    except:
        print("Invalid input!")
        return None

    # Get year
    print("Enter year (e.g., 2026): ", end="")
    try:
        year = int(input())
        if year < 2015:
            print(
                "Warning: Model trained on 2015-2024 data. Results may be less accurate."
            )
    except:
        print("Invalid input!")
        return None

    return commodity, city_name, lat, lon, month, year


def predict(commodity, city_name, lat, lon, month, year):
    # Encode
    commodity_encoded = le_commodity.transform([commodity])[0]
    year_trend = year - 2015

    # Season
    if month in [6, 7, 8, 9]:
        season = 0  # Monsoon
    elif month in [10, 11]:
        season = 1  # Post-monsoon
    elif month in [12, 1, 2]:
        season = 2  # Winter
    else:
        season = 3  # Summer

    is_monsoon = 1 if month in [6, 7, 8, 9] else 0

    # Get LIVE weather
    print(f"\n  Fetching live weather for {city_name}...")
    rainfall_mm = get_live_weather(lat, lon)

    if rainfall_mm is None:
        # Fallback to typical rainfall
        rainfall_map = {0: 200, 1: 150, 2: 30, 3: 50}
        rainfall_mm = rainfall_map[season]
        print(f"  Using typical rainfall: {rainfall_mm}mm")

    rainfall_deviation = ((rainfall_mm - 50) / 50) * 100
    rainfall_category = 0 if rainfall_mm < 50 else (1 if rainfall_mm < 150 else 2)
    # Get LIVE Macro features
    print(f"  Fetching live macro-economic indicators...")
    petrol_price = live_macro_fetcher.fetch_live_petrol_price(year, month)
    tension_score, headlines = live_macro_fetcher.fetch_live_geopolitical_news()

    features = [
        [
            commodity_encoded,
            year_trend,
            month,
            season,
            is_monsoon,
            rainfall_mm,
            rainfall_deviation,
            rainfall_category,
            petrol_price,
            tension_score
        ]
    ]

    base_price = model_gb.predict(features)[0] if model_gb else 0.0
    
    # We remove the 6.5% inflation rate extrapolation here because our new model 
    # uses the petrol_price feature to dynamically learn the inflation rate!
    price = base_price

    season_names = ["Monsoon", "Post-monsoon", "Winter", "Summer"]

    print("\n" + "=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    print(f"  Commodity:     {commodity}")
    print(f"  City:          {city_name}, Tamil Nadu")
    print(f"  Month:         {month}/{year}")
    print(f"  Season:        {season_names[season]}")
    print(f"  Rainfall:      {rainfall_mm:.0f}mm")
    print(f"  Petrol Price:  Rs. {petrol_price}")
    print(f"  Geopolitics:   {tension_score}/100 Tension Index")
    print(f"  Top News:      {headlines[0] if headlines else 'None'}")
    print("=" * 50)
    print(f"  PREDICTED PRICE: Rs. {price:.2f} per kg")
    print("=" * 50)


def main():
    while True:
        show_menu()

        inputs = get_user_input()
        if inputs:
            commodity, city_name, lat, lon, month, year = inputs
            predict(commodity, city_name, lat, lon, month, year)

        print("\nPredict again? (y/n): ", end="")
        if input().lower() != "y":
            print("\nThankyou!")
            break


if __name__ == "__main__":
    main()
