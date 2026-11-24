import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AQI Prediction", page_icon="🌍", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "aqi_environmental_data.csv"
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    X, y = data[FEATURES], data["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred)
    }
    return model, metrics, X_test, y_test, pred

def category(value):
    if value <= 50: return "Good", "🟢"
    if value <= 100: return "Satisfactory", "🟡"
    if value <= 200: return "Moderate", "🟠"
    if value <= 300: return "Poor", "🔴"
    if value <= 400: return "Very Poor", "🟣"
    return "Severe", "🟥"

st.title("🌍 Air Quality Index (AQI) Prediction")
model, metrics, X_test, y_test, predictions = train_model()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a section", ["Dashboard", "Predict AQI", "Dataset Explorer", "Historical Country AQI", "Model Performance"])
    st.markdown("---")

data = load_data()
if page == "Dashboard":
    st.subheader("Environmental Data Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Average AQI", f"{data.AQI.mean():.1f}")
    c3.metric("Maximum AQI", f"{data.AQI.max():.1f}")
    c4.metric("Model R²", f"{metrics['r2']:.3f}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### AQI distribution")
        fig, ax = plt.subplots()
        ax.hist(data["AQI"], bins=30)
        ax.set_xlabel("AQI"); ax.set_ylabel("Number of records")
        st.pyplot(fig)
    with right:
        st.markdown("#### Average pollutant levels")
        pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        fig, ax = plt.subplots()
        data[pollutants].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Average measured value")
        st.pyplot(fig)
    st.markdown("#### Sample records")
    st.dataframe(data.head(10), use_container_width=True)

elif page == "Predict AQI":
    st.subheader("Enter environmental measurements")
    st.write("Enter pollutant concentrations and weather conditions to estimate AQI.")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            pm25_v = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 45.0)
            pm10_v = st.number_input("PM10 (µg/m³)", 0.0, 600.0, 90.0)
            no2_v = st.number_input("NO₂ (µg/m³)", 0.0, 300.0, 25.0)
            so2_v = st.number_input("SO₂ (µg/m³)", 0.0, 200.0, 10.0)
        with b:
            co_v = st.number_input("CO (mg/m³)", 0.0, 30.0, 1.0)
            o3_v = st.number_input("O₃ (µg/m³)", 0.0, 300.0, 50.0)
            temp_v = st.number_input("Temperature (°C)", -5.0, 55.0, 29.0)
            humidity_v = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        submitted = st.form_submit_button("Predict AQI", type="primary")
    if submitted:
        row = pd.DataFrame([[pm25_v, pm10_v, no2_v, so2_v, co_v, o3_v, temp_v, humidity_v]], columns=FEATURES)
        result = float(model.predict(row)[0])
        label, emoji = category(result)
        st.markdown("---")
        x, y = st.columns(2)
        x.metric("Predicted AQI", f"{result:.1f}")
        y.metric("Air Quality Category", f"{emoji} {label}")
        st.info("This is a machine-learning estimate based on the supplied training dataset. It is not an official CPCB reading or health advisory.")

elif page == "Dataset Explorer":
    st.subheader("Explore the training dataset")
    selected = st.multiselect("Select columns", data.columns.tolist(), default=["PM2.5", "PM10", "NO2", "AQI"])
    st.dataframe(data[selected].describe().round(2), use_container_width=True)
    st.dataframe(data[selected].head(100), use_container_width=True)
    st.download_button("Download dataset CSV", data.to_csv(index=False), "aqi_environmental_data.csv", "text/csv")
    st.markdown("#### Correlation with AQI")
    corr = data.corr(numeric_only=True)["AQI"].drop("AQI").sort_values()
    fig, ax = plt.subplots()
    corr.plot(kind="barh", ax=ax)
    ax.set_xlabel("Correlation")
    st.pyplot(fig)


elif page == "Historical Country AQI":
    st.subheader("Historical AQI by Country")
    history_path = Path(__file__).parent / "data" / "country_aqi_history.csv"
    history = pd.read_csv(history_path)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")

    c1, c2, c3 = st.columns(3)
    c1.metric("Historical records", f"{len(history):,}")
    c2.metric("Countries", f"{history['Country'].nunique():,}")
    c3.metric("Average AQI", f"{history['AQI Value'].mean():.1f}")

    countries = sorted(history["Country"].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Choose countries", countries,
        default=countries[:min(5, len(countries))]
    )
    filtered = history[history["Country"].isin(selected_countries)] if selected_countries else history.iloc[0:0]

    if not filtered.empty:
        st.markdown("#### AQI trend over time")
        trend = filtered.groupby(["Date", "Country"], as_index=False)["AQI Value"].mean()
        pivot = trend.pivot(index="Date", columns="Country", values="AQI Value")
        st.line_chart(pivot)

        st.markdown("#### Average AQI by selected country")
        averages = filtered.groupby("Country")["AQI Value"].mean().sort_values(ascending=False)
        st.bar_chart(averages)
    else:
        st.info("Select at least one country to display its historical data.")

    st.markdown("#### Filter records")
    statuses = sorted(history["Status"].dropna().unique().tolist())
    selected_statuses = st.multiselect("Filter by AQI status", statuses, default=statuses)
    table = filtered[filtered["Status"].isin(selected_statuses)] if selected_countries else history.iloc[0:0]
    st.dataframe(table, use_container_width=True, height=350)
    st.download_button(
        "Download filtered historical data",
        table.to_csv(index=False),
        "filtered_country_aqi_history.csv",
        "text/csv"
    )

else:
    st.subheader("Model evaluation")
    a, b, c = st.columns(3)
    a.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
    b.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
    c.metric("R² Score", f"{metrics['r2']:.3f}")
    compare = pd.DataFrame({"Actual AQI": y_test.values, "Predicted AQI": predictions})
    st.dataframe(compare.head(20).round(1), use_container_width=True)
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    low = min(y_test.min(), predictions.min()); high = max(y_test.max(), predictions.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
    ax.set_title("Actual vs Predicted AQI")
    st.pyplot(fig)
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance")
    st.pyplot(fig)

st.markdown("---")
