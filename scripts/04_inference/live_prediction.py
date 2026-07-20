"""
LIVE WEATHER PRICE PREDICTION
Uses Open-Meteo API for real-time weather data
Predicts Tamil Nadu crop prices with live weather
"""

import requests
import joblib
from datetime import datetime
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Tamil Nadu major cities coordinates
TN_LOCATIONS = {
    "Chennai": (13.0827, 80.2707),
    "Madurai": (9.9252, 78.1198),
    "Coimbatore": (11.0168, 76.9558),
    "Thanjavur": (10.7870, 79.1378),
    "Salem": (11.6643, 78.1460),
    "Trichy": (10.7905, 78.7047),
}


def get_live_weather(city="Chennai"):
    """Fetch live weather from Open-Meteo API (FREE, no key needed!)."""

    lat, lon = TN_LOCATIONS.get(city, TN_LOCATIONS["Chennai"])

    # Open-Meteo API (free, no API key required)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "daily" in data:
            daily = data["daily"]
            return {
                "dates": daily["time"],
                "precipitation": daily["precipitation_sum"],
                "temp_max": daily["temperature_2m_max"],
                "temp_min": daily["temperature_2m_min"],
                "city": city,
            }
    except Exception as e:
        print(f"  API Error: {e}")
        return None

    return None


def predict_with_live_weather(commodity="Rice", city="Chennai"):
    """Predict price using live weather data."""

    print("=" * 60)
    print("LIVE WEATHER PRICE PREDICTION")
    print("=" * 60)

    # Get live weather
    print(f"\n  Fetching live weather for {city}...")
    weather = get_live_weather(city)

    if not weather:
        print("  Could not fetch weather. Using default values.")
        rainfall_mm = 20
        rainfall_deviation = 0
    else:
        # Calculate monthly rainfall estimate from 7-day forecast
        weekly_rain = sum([p for p in weather["precipitation"] if p])
        rainfall_mm = weekly_rain * 4  # Estimate monthly from weekly

        # Typical Feb rainfall in TN is ~20mm
        typical_rain = 20
        rainfall_deviation = ((rainfall_mm - typical_rain) / typical_rain) * 100

        print(f"\n  LIVE WEATHER DATA ({city}):")
        print("  ----------------------------------------")
        for i, date in enumerate(weather["dates"][:5]):
            rain = weather["precipitation"][i] or 0
            temp_max = weather["temp_max"][i]
            temp_min = weather["temp_min"][i]
            print(f"    {date}: Rain={rain}mm, Temp={temp_min}-{temp_max}C")
        print("  ----------------------------------------")
        print(f"    Weekly rainfall: {weekly_rain:.1f}mm")
        print(f"    Monthly estimate: {rainfall_mm:.1f}mm")
        print(f"    Deviation from normal: {rainfall_deviation:.1f}%")

    # Load model
    print("\n  Loading prediction model...")
    model_data = joblib.load(MODELS_DIR / "tn_no_lag_model.joblib")
    model = model_data["model"]
    le_commodity = model_data["le_commodity"]

    # Check commodity
    if commodity not in le_commodity.classes_:
        print(
            f"  Error: '{commodity}' not found. Available: {list(le_commodity.classes_)}"
        )
        return None

    # Prepare features
    today = datetime.now()
    commodity_encoded = le_commodity.transform([commodity])[0]
    year_trend = today.year - 2015
    month = today.month

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

    # Rainfall category
    if rainfall_mm < 50:
        rainfall_category = 0
    elif rainfall_mm < 150:
        rainfall_category = 1
    elif rainfall_mm < 300:
        rainfall_category = 2
    else:
        rainfall_category = 3

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
        ]
    ]

    # Predict
    prediction = model.predict(features)[0]

    # Display result
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"    Commodity: {commodity}")
    print(f"    Location:  {city}, Tamil Nadu")
    print(f"    Date:      {today.strftime('%B %d, %Y')}")
    print(f"    Weather:   {rainfall_mm:.0f}mm rain expected")
    print("=" * 60)
    print(f"    PREDICTED PRICE: Rs. {prediction:.2f} per kg")
    print("=" * 60)

    return prediction


def predict_multiple_commodities(city="Chennai"):
    """Predict prices for all major commodities."""

    print("\n" + "=" * 60)
    print(f"LIVE PREDICTIONS FOR {city.upper()}, TAMIL NADU")
    print("=" * 60)

    # Get weather once
    weather = get_live_weather(city)
    if weather:
        weekly_rain = sum([p for p in weather["precipitation"] if p])
        rainfall_mm = weekly_rain * 4
        rainfall_deviation = ((rainfall_mm - 20) / 20) * 100
        print(f"  Weather: {weekly_rain:.1f}mm rain this week")
    else:
        rainfall_mm = 20
        rainfall_deviation = 0
        print("  Weather: Using defaults")

    # Load model
    model_data = joblib.load(MODELS_DIR / "tn_no_lag_model.joblib")
    model = model_data["model"]
    le_commodity = model_data["le_commodity"]

    today = datetime.now()
    year_trend = today.year - 2015
    month = today.month
    season = 2 if month in [12, 1, 2] else (0 if month in [6, 7, 8, 9] else 3)
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0
    rainfall_category = 0 if rainfall_mm < 50 else (1 if rainfall_mm < 150 else 2)

    # Common commodities
    commodities = ["Rice", "Wheat", "Tomato", "Onion", "Potato", "Milk", "Sugar"]

    print(f"\n  {'Commodity':<15} {'Predicted Price':>15}")
    print("  " + "-" * 35)

    for commodity in commodities:
        if commodity in le_commodity.classes_:
            commodity_encoded = le_commodity.transform([commodity])[0]
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
                ]
            ]
            price = model.predict(features)[0]
            print(f"  {commodity:<15} Rs. {price:>10.2f}/kg")

    print("  " + "-" * 35)
    print(f"  Date: {today.strftime('%B %d, %Y')}")


if __name__ == "__main__":
    # Single commodity prediction
    predict_with_live_weather("Tomato", "Madurai")

    print("\n")

    # Multiple commodities
    predict_multiple_commodities("Chennai")
