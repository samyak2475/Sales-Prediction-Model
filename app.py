import streamlit as st
import numpy as np
import joblib

# Load trained model
regressor = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="BigMart Sales Prediction")

st.title("🛒 BigMart Sales Prediction App")
st.write("Enter product and outlet details")

# ---------- INPUTS ----------
item_identifier = st.number_input("Item Identifier (encoded)", min_value=0)
item_weight = st.number_input("Item Weight", min_value=0.0)
item_fat = st.number_input("Item Fat Content (encoded)", min_value=0)
item_visibility = st.number_input("Item Visibility", min_value=0.0)
item_type = st.number_input("Item Type (encoded)", min_value=0)
item_mrp = st.number_input("Item MRP", min_value=0.0)

outlet_identifier = st.number_input("Outlet Identifier (encoded)", min_value=0)
outlet_year = st.number_input("Outlet Establishment Year", min_value=1900)
outlet_size = st.number_input("Outlet Size (encoded)", min_value=0)
outlet_location = st.number_input("Outlet Location Type (encoded)", min_value=0)
outlet_type = st.number_input("Outlet Type (encoded)", min_value=0)

# ---------- PREDICTION ----------
if st.button("Predict Sales"):
    input_data = np.array([[
        item_identifier,
        item_weight,
        item_fat,
        item_visibility,
        item_type,
        item_mrp,
        outlet_identifier,
        outlet_year,
        outlet_size,
        outlet_location,
        outlet_type
    ]])

    prediction = regressor.predict(input_data)

    st.success(f"💰 Predicted Sales: ₹ {prediction[0]:,.2f}")
