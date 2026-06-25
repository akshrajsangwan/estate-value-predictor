import streamlit as st
import pandas as pd
import joblib
from custom_transformers import *

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("EstateValuePredictor.pkl")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🏠 California House Price Predictor")
st.markdown("Predict California house prices using Machine Learning.")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    This application predicts house prices using
    a Machine Learning model trained on the
    California Housing Dataset.

    Features:
    - Latitude
    - Longitude
    - Housing Median Age
    - Median Income
    - Total Rooms
    - Total Bedrooms
    - Population
    - Households
    - Median House Value
    - Location
    - Ocean Proximity
    """
)

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-113.0,
        value=-122.23
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=37.88
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1,
        max_value=100,
        value=41
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        max_value=20.0,
        value=8.3252
    )

with col2:
    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=880
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=129
    )

    population = st.number_input(
        "Population",
        min_value=1,
        value=322
    )

    households = st.number_input(
        "Households",
        min_value=1,
        value=126
    )

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict House Price", use_container_width=True):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    try:
        prediction = model.predict(input_data)

        st.success("Prediction Successful ✅")

        st.subheader("🏡 Estimated House Value")

        st.metric(
            label="Predicted Price",
            value=f"${prediction[0]:,.2f}"
        )

        st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

st.caption(
    "Developed by Akshraj Sangwan | B.Tech CSE-AIML | California Estate Value Predictor Project"
)