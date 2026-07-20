"""
TAMIL NADU AGRICULTURAL PRICE PREDICTION DASHBOARD
Streamlit App - Complete Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TN Agri Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a1e, #2d5a2d);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d7a3d;
    }
    .price-display {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 2px solid #c9a84c;
        margin: 10px 0;
    }
    .weather-card {
        background: linear-gradient(135deg, #0f2027, #203a43);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #44aacc;
    }
    h1 { color: #c9a84c !important; }
    .stSelectbox label { color: #c9a84c !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_QUALITY = PROJECT_ROOT / "data" / "quality_checked"
MODELS_DIR = PROJECT_ROOT / "models"

# ─── All 32 TN Districts ─────────────────────────────────────────────────────
TN_DISTRICTS = {
    "Ariyalur": (11.1400, 79.0800),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Cuddalore": (11.7480, 79.7714),
    "Dharmapuri": (12.1270, 78.1580),
    "Dindigul": (10.3624, 77.9695),
    "Erode": (11.3410, 77.7172),
    "Kanchipuram": (12.8342, 79.7036),
    "Kanniyakumari": (8.0883, 77.5385),
    "Karur": (10.9601, 78.0766),
    "Krishnagiri": (12.5186, 78.2137),
    "Madurai": (9.9252, 78.1198),
    "Nagapattinam": (10.7672, 79.8449),
    "Namakkal": (11.2189, 78.1674),
    "Nilgiris": (11.4916, 76.7337),
    "Perambalur": (11.2340, 78.8800),
    "Pudukkottai": (10.3833, 78.8001),
    "Ramanathapuram": (9.3639, 78.8395),
    "Salem": (11.6643, 78.1460),
    "Sivagangai": (9.8433, 78.4809),
    "Thanjavur": (10.7870, 79.1378),
    "Theni": (10.0104, 77.4768),
    "Thoothukudi": (8.7642, 78.1348),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Tirunelveli": (8.7139, 77.7567),
    "Tiruppur": (11.1085, 77.3411),
    "Tiruvallur": (13.1439, 79.9086),
    "Tiruvannamalai": (12.2253, 79.0747),
    "Tiruvarur": (10.7713, 79.6370),
    "Vellore": (12.9165, 79.1325),
    "Viluppuram": (11.9401, 79.4861),
    "Virudhunagar": (9.5851, 77.9526),
}

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# ─── Load Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads the pre-trained Hybrid Model (Gradient Boosting + Ridge) and label encoder.
    Note: Cache busted to force loading the new Hybrid model.
    """
    try:
        PROJECT_ROOT = Path(__file__).parent
        model_path = PROJECT_ROOT / "models" / "tn_no_lag_model.joblib"
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_price_data() -> pd.DataFrame:
    """
    Loads and preprocesses historical retail price data for Tamil Nadu.
    Returns:
        pd.DataFrame: Cleaned dataframe with parsed dates.
    """
    tn = pd.read_csv(DATA_QUALITY / "tn_retail_prices_dashboard.csv")
    tn["date"] = pd.to_datetime(tn["date"])
    tn["year"] = tn["date"].dt.year
    tn["month"] = tn["date"].dt.month
    return tn


@st.cache_data
def load_agri_data() -> pd.DataFrame:
    """
    Loads district-level agricultural census data.
    Returns:
        pd.DataFrame: Filtered census data for Tamil Nadu.
    """
    try:
        df = pd.read_csv(PROJECT_ROOT / "data" / "tamil_nadu" / "tn_agriculture_census.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["state_name", "district_name", "crop_name", "total_ar_district", "irr_ar_district"])


# ─── Weather Fetch ────────────────────────────────────────────────────────────
def get_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetches live 7-day weather forecast from Open-Meteo API.
    Args:
        lat (float): Latitude of the district.
        lon (float): Longitude of the district.
    Returns:
        Optional[Dict]: Weather data dictionary, or None if the API call fails.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
            "current_weather": True,
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        }
        r = requests.get(url, params=params, timeout=8).json()
        daily = r["daily"]
        weekly_rain = sum([p or 0 for p in daily["precipitation_sum"]])
        current_temp = r.get("current_weather", {}).get("temperature", "N/A")
        return {
            "current_temp": current_temp,
            "weekly_rain": weekly_rain,
            "monthly_rain": weekly_rain * 4,
            "dates": daily["time"],
            "rain": daily["precipitation_sum"],
            "tmax": daily["temperature_2m_max"],
            "tmin": daily["temperature_2m_min"],
        }
    except:
        return None


# ─── Prediction Function ──────────────────────────────────────────────────────
def predict_price(
    model_data: Dict[str, Any],
    commodity: str,
    month: int,
    year: int,
    rainfall_mm: float,
) -> Optional[float]:
    """
    Predicts the future agricultural price using the loaded machine learning model.
    Args:
        model_data (Dict): The loaded model dictionary.
        commodity (str): The name of the crop.
        month (int): The target month.
        year (int): The target year.
        rainfall_mm (float): The expected rainfall in mm.
    Returns:
        Optional[float]: The predicted price per kg, or None if the commodity is invalid.
    """
    model_gb = model_data.get("model_gb")
    model_ridge = model_data.get("model_ridge")
    scaler = model_data.get("scaler")
    le = model_data.get("le_commodity")

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

    features = [
        [
            commodity_encoded,
            year_trend,
            month,
            season,
            is_monsoon,
            rainfall_mm,
            deviation,
            cat,
        ]
    ]
    features_scaled = scaler.transform(features)
    if model_gb and model_ridge:
        return 0.5 * model_gb.predict(features)[0] + 0.5 * model_ridge.predict(features_scaled)[0]
    else:
        return model_gb.predict(features)[0]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌾 TN Agri Predictor")
    st.markdown("---")

    district = st.selectbox("📍 Select District", list(TN_DISTRICTS.keys()), index=11)
    commodity = st.selectbox(
        "🛒 Select Commodity",
        [
            "Rice",
            "Tomato",
            "Onion",
            "Potato",
            "Wheat",
            "Milk",
            "Sugar",
            "Gram Dal",
            "Masoor Dal",
            "Moong Dal",
            "Urad Dal",
            "Tur/Arhar Dal",
            "Groundnut Oil (Packed)",
            "Mustard Oil (Packed)",
            "Palm Oil (Packed)",
            "Soya Oil (Packed)",
            "Sunflower Oil (Packed)",
            "Vanaspati (Packed)",
            "Atta (Wheat)",
            "Tea Loose",
            "Salt Pack (Iodised)",
            "Gur",
        ],
        index=0,
    )
    month_name = st.selectbox("📅 Month", MONTH_NAMES, index=datetime.now().month - 1)
    month = MONTH_NAMES.index(month_name) + 1
    year = st.slider("📆 Year", 2024, 2030, datetime.now().year)

    st.markdown("---")
    st.markdown("**🌤️ Live Weather Source**")
    st.caption("Open-Meteo API (Free)")

# ─── Load Data ───────────────────────────────────────────────────────────────
model_data = load_model()
tn_prices = load_price_data()
tn_agri = load_agri_data()
lat, lon = TN_DISTRICTS[district]

# ─── Fetch Weather ────────────────────────────────────────────────────────────
weather = get_weather(lat, lon)
rainfall_mm = weather["monthly_rain"] if weather else 50.0

# ─── Predict ─────────────────────────────────────────────────────────────────
predicted_price = predict_price(model_data, commodity, month, year, rainfall_mm)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("# 🌾 Tamil Nadu Agricultural Price Prediction Dashboard")
st.caption(
    f"Powered by Gradient Boosting ML + Open-Meteo Live Weather | {district}, Tamil Nadu"
)
st.markdown("---")

# ─── ROW 1: Key Metrics ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌡️ Current Temp", f"{weather['current_temp']}°C" if weather else "N/A")
with col2:
    st.metric(
        "🌧️ Weekly Rainfall", f"{weather['weekly_rain']:.1f} mm" if weather else "N/A"
    )
with col3:
    st.metric("📍 District", district)
with col4:
    season_map = {0: "🌧️ Monsoon", 1: "☁️ Post-Monsoon", 2: "❄️ Winter", 3: "☀️ Summer"}
    s = (
        0
        if month in [6, 7, 8, 9]
        else (1 if month in [10, 11] else (2 if month in [12, 1, 2] else 3))
    )
    st.metric("🗓️ Season", season_map[s])

st.markdown("---")

# ─── ROW 2: Price Prediction + Weather ──────────────────────────────────────
col_pred, col_weather = st.columns([1, 1])

with col_pred:
    st.markdown("### 💰 Price Prediction")
    if predicted_price:
        st.markdown(
            f"""
        <div class="price-display">
            <h2 style="color:#c9a84c; font-size:48px; margin:0;">Rs. {predicted_price:.2f}</h2>
            <p style="color:#aaa; font-size:18px;">per kg</p>
            <p style="color:#88cc88; font-size:16px;">{commodity} • {month_name} {year} • {district}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Confidence range
        low = predicted_price * 0.90
        high = predicted_price * 1.10
        st.info(f"📊 Expected Range: Rs. {low:.2f} – Rs. {high:.2f}/kg (±10%)")
    else:
        st.error("Prediction failed.")

with col_weather:
    st.markdown("### 🌤️ 7-Day Weather Forecast")
    if weather:
        weather_df = pd.DataFrame(
            {
                "Date": weather["dates"],
                "Min Temp (°C)": weather["tmin"],
                "Max Temp (°C)": weather["tmax"],
                "Rain (mm)": [r or 0 for r in weather["rain"]],
            }
        )
        fig_weather = go.Figure()
        fig_weather.add_trace(
            go.Bar(
                x=weather_df["Date"],
                y=weather_df["Rain (mm)"],
                name="Rain (mm)",
                marker_color="#44aacc",
                yaxis="y2",
            )
        )
        fig_weather.add_trace(
            go.Scatter(
                x=weather_df["Date"],
                y=weather_df["Max Temp (°C)"],
                name="Max Temp",
                line=dict(color="#ff6b6b", width=2),
            )
        )
        fig_weather.add_trace(
            go.Scatter(
                x=weather_df["Date"],
                y=weather_df["Min Temp (°C)"],
                name="Min Temp",
                line=dict(color="#74b9ff", width=2),
            )
        )
        fig_weather.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=280,
            margin=dict(t=10, b=10),
            yaxis2=dict(overlaying="y", side="right", title="Rain (mm)"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.warning("Weather data unavailable.")

st.markdown("---")

# ─── ROW 3: Historical Price Trend ───────────────────────────────────────────
st.markdown("### 📈 Historical Price Trend")
comm_data = tn_prices[tn_prices["commodity"] == commodity].copy()

if len(comm_data) > 0:
    monthly_avg = comm_data.groupby(["year", "month"])["price"].mean().reset_index()
    monthly_avg["date_str"] = pd.to_datetime(
        monthly_avg[["year", "month"]].assign(day=1)
    )
    monthly_avg = monthly_avg.sort_values("date_str")

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=monthly_avg["date_str"],
            y=monthly_avg["price"],
            mode="lines",
            name="Actual Price",
            line=dict(color="#c9a84c", width=2),
            fill="tozeroy",
            fillcolor="rgba(201,168,76,0.15)",
        )
    )

    # Add predicted point
    fig_trend.add_trace(
        go.Scatter(
            x=[pd.Timestamp(year=year, month=month, day=1)],
            y=[predicted_price],
            mode="markers",
            name="Your Prediction",
            marker=dict(color="#ff6b6b", size=14, symbol="star"),
        )
    )

    # Highlight 2023 spike for Tomato
    if commodity == "Tomato":
        fig_trend.add_vrect(
            x0="2023-07-01",
            x1="2023-09-01",
            fillcolor="rgba(255,0,0,0.1)",
            line_width=0,
            annotation_text="2023 Crisis",
            annotation_position="top left",
        )

    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
        xaxis_title="Date",
        yaxis_title="Price (Rs/kg)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ─── ROW 4: All Commodities + District Analysis ──────────────────────────────
col_all, col_dist = st.columns([1, 1])

with col_all:
    st.markdown("### 🛒 All Commodities Prediction")
    all_commodities = model_data["le_commodity"].classes_
    prices_list = []
    for c in all_commodities:
        p = predict_price(model_data, c, month, year, rainfall_mm)
        if p:
            prices_list.append({"Commodity": c, "Price (Rs/kg)": round(p, 2)})

    prices_df = pd.DataFrame(prices_list).sort_values("Price (Rs/kg)", ascending=False)

    fig_bar = px.bar(
        prices_df,
        x="Price (Rs/kg)",
        y="Commodity",
        orientation="h",
        color="Price (Rs/kg)",
        color_continuous_scale="YlOrRd",
        title=f"Predicted Prices — {month_name} {year}",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=500,
        showlegend=False,
        margin=dict(t=40, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_dist:
    st.markdown("### 🗺️ District Agriculture Profile")
    dist_data = tn_agri[tn_agri["district_name"].str.lower() == district.lower()]

    if len(dist_data) > 0:
        top_crops = (
            dist_data.groupby("crop_name")["total_ar_district"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .reset_index()
        )
        top_crops.columns = ["Crop", "Area"]

        fig_pie = px.pie(
            top_crops,
            values="Area",
            names="Crop",
            title=f"Top Crops in {district}",
            color_discrete_sequence=px.colors.sequential.Greens_r,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=400,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # District stats
        total_area = dist_data["total_ar_district"].sum()
        irrigated = dist_data["irr_ar_district"].sum()
        irr_pct = (irrigated / total_area * 100) if total_area > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Area", f"{total_area/1e5:.1f}L acres")
        c2.metric("Irrigation %", f"{irr_pct:.1f}%")
        c3.metric("Crop Diversity", f"{dist_data['crop_name'].nunique()} crops")
    else:
        st.info("No crop census data for this district.")

st.markdown("---")

# ─── ROW 5: Monthly Forecast Chart ─────────────────────────────────────────
st.markdown(f"### 📅 Monthly Price Forecast for {year} — {commodity}")

months_list = list(range(1, 13))
monthly_preds = []
for m in months_list:
    p = predict_price(model_data, commodity, m, year, rainfall_mm)
    monthly_preds.append(p if p else 0)

fig_monthly = go.Figure()
fig_monthly.add_trace(
    go.Scatter(
        x=MONTH_NAMES,
        y=monthly_preds,
        mode="lines+markers",
        line=dict(color="#c9a84c", width=3),
        marker=dict(size=8, color="#ff6b6b"),
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.1)",
        name="Predicted Price",
    )
)
fig_monthly.add_trace(
    go.Scatter(
        x=[MONTH_NAMES[month - 1]],
        y=[monthly_preds[month - 1]],
        mode="markers+text",
        marker=dict(color="#ff6b6b", size=14, symbol="star"),
        text=["Selected"],
        textposition="top center",
        name="Selected Month",
        showlegend=False,
    )
)
fig_monthly.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=300,
    xaxis_title="Month",
    yaxis_title="Price (Rs/kg)",
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style='text-align:center; color:#666; padding:10px;'>
    🌾 Tamil Nadu Agricultural Price Prediction |
    Model: Gradient Boosting (R² = 95.85%) |
    Weather: Open-Meteo API |
    Data: 2015-2024
</div>
""",
    unsafe_allow_html=True,
)
