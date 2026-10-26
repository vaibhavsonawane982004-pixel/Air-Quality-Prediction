import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/best_co_model.pkl")

    with open("models/model_info.json", "r") as file:
        info = json.load(file)

    return model, info

model, info = load_model()

st.title("🌍 Air Quality Prediction System")
st.subheader("Carbon Monoxide (CO) Concentration Estimation")

st.write(
    "Enter the available sensor, pollutant, and environmental "
    "measurements to estimate CO concentration."
)

st.info(
    "This model estimates CO concentration from contemporaneous "
    "measurements. It is not a forecast of future pollution."
)

with st.sidebar:
    st.header("About the Model")
    st.write(f"**Model:** {info['model_name']}")
    st.write(f"**Target:** CO concentration")
    st.write(f"**MAE:** {info['mae']:.4f} mg/m³")
    st.write(f"**RMSE:** {info['rmse']:.4f} mg/m³")
    st.write(f"**R²:** {info['r2']:.4f}")

st.header("Enter Input Measurements")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pt08_s1 = st.number_input(
            "PT08.S1(CO) sensor response",
            value=1000.0
        )

        c6h6 = st.number_input(
            "Benzene — C6H6(GT)",
            value=10.0
        )

        pt08_s2 = st.number_input(
            "PT08.S2(NMHC) sensor response",
            value=900.0
        )

        nox = st.number_input(
            "NOx(GT)",
            value=150.0
        )

        no2 = st.number_input(
            "NO2(GT)",
            value=100.0
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            value=20.0
        )

        humidity = st.number_input(
            "Relative Humidity (%)",
            value=50.0
        )

        absolute_humidity = st.number_input(
            "Absolute Humidity",
            value=1.0
        )

        hour = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=12
        )

        month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=6
        )

    submitted = st.form_submit_button("Predict CO Concentration")

if submitted:
    input_data = pd.DataFrame([{
        "PT08.S1(CO)": pt08_s1,
        "C6H6(GT)": c6h6,
        "PT08.S2(NMHC)": pt08_s2,
        "NOx(GT)": nox,
        "NO2(GT)": no2,
        "T": temperature,
        "RH": humidity,
        "AH": absolute_humidity,
        "Hour": hour,
        "Month": month
    }])

    prediction = float(model.predict(input_data)[0])

    st.header("Prediction Result")

    st.metric(
        label="Estimated CO Concentration",
        value=f"{prediction:.3f} mg/m³"
    )

    if prediction < 0:
        st.warning(
            "The model produced a negative estimate. "
            "Treat this as a model limitation; actual CO "
            "concentration cannot be negative."
        )

    st.caption(
        "This is a model estimate based on the inputs provided, "
        "not an official air-quality or health-safety assessment."
    )
