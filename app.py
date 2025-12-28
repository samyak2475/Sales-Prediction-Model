import streamlit as st
import numpy as np
import joblib

# Load trained XGBoost model
regressor = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="BigMart Sales Predictor")

st.title("🛒 BigMart Sales Prediction App")
st.write("Enter product and outlet details to predict sales")

# ---- INPUTS ----
item_mrp = st.number_input("Item MRP", min_value=0.0)
item_visibility = st.number_input("Item Visibility", min_value=0.0)
item_weight = st.number_input("Item Weight", min_value=0.0)

outlet_age = st.number_input("Outlet Age (Years)", min_value=0)

outlet_size = st.selectbox(
    "Outlet Size",
    ["Small", "Medium", "High"]
)

outlet_location = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

# ---- ENCODING ----
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
outlet_location_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}

outlet_size_encoded = outlet_size_map[outlet_size]
outlet_location_encoded = outlet_location_map[outlet_location]

# ---- PREDICTION ----
if st.button("Predict Sales"):
    input_data = np.array([[  
        item_mrp,
        item_visibility,
        item_weight,
        outlet_age,
        outlet_size_encoded,
        outlet_location_encoded
    ]])

    prediction = regressor.predict(input_data)

    st.success(f"💰 Predicted Sales: ₹ {prediction[0]:,.2f}")
