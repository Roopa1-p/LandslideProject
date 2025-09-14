import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and feature info
model_obj = joblib.load("landslide_rf_model.joblib")
model = model_obj["model"]
scaler = model_obj["scaler"]
features = model_obj["features"]
num_features = model_obj["num_features"]

st.set_page_config(page_title="🌍 Landslide Susceptibility Prediction", layout="wide")

st.title("🌍 AI-Powered Landslide Susceptibility Prediction")
st.write("Predict landslide risk in Himalayan & Western Ghats regions using environmental and terrain factors.")

# Sidebar inputs (numeric features)
st.sidebar.header("🌐 Enter Environmental Data")
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
slope = st.sidebar.number_input("Slope Angle (degrees)", min_value=0.0, step=0.1)
soil_saturation = st.sidebar.number_input("Soil Saturation (%)", min_value=0.0, max_value=100.0, step=0.1)
vegetation = st.sidebar.number_input("Vegetation Cover (%)", min_value=0.0, max_value=100.0, step=0.1)
proximity_water = st.sidebar.number_input("Proximity to Water (km)", min_value=0.0, step=0.1)
earthquake = st.sidebar.number_input("Earthquake Activity (magnitude)", min_value=0.0, step=0.1)

# Main screen dropdown for Soil Type
st.subheader("🧱 Select Soil Type")
soil_type = st.selectbox("Soil Type", ["Gravel", "Sand", "Silt"])

# Build input row
row = {
    'Rainfall_mm': rainfall,
    'Slope_Angle': slope,
    'Soil_Saturation': soil_saturation,
    'Vegetation_Cover': vegetation,
    'Proximity_to_Water': proximity_water,
    'Earthquake_Activity': earthquake,
    'Soil_Type_Gravel': 0,
    'Soil_Type_Sand': 0,
    'Soil_Type_Silt': 0
}
row[f"Soil_Type_{soil_type}"] = 1

# Ensure correct order
input_data = pd.DataFrame([[row[f] for f in features]], columns=features)

# Predict for single input
if st.button("🔮 Predict Landslide Risk"):
    input_data[num_features] = scaler.transform(input_data[num_features])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("✅ Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Landslide! (Probability: {probability:.2f})")
    else:
        st.success(f"🌱 Low Risk of Landslide. (Probability: {probability:.2f})")

# Bulk prediction from CSV
st.subheader("📂 Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Landslide" in df.columns:
        df = df.drop(columns=["Landslide"])

    # Add missing features if not present
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Reorder and scale
    df = df[features]
    df[num_features] = scaler.transform(df[num_features])

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["Landslide_Risk"] = preds
    df["Risk_Probability"] = probs

    st.success("✅ Bulk prediction complete!")
    st.dataframe(df.head())

    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_out,
        file_name="landslide_predictions.csv",
        mime="text/csv"
    )
