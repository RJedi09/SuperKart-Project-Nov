# Import os to locate the saved model.
import os
# Import Streamlit to create the prediction app.
import streamlit as st
# Import pandas to build the prediction input.
import pandas as pd
# Import joblib to load the trained model.
import joblib
# Set the location of the saved model.
model_path = os.path.join(os.path.dirname(__file__), "best_superkart_model_v1.joblib")
# Load the model trained by GitHub Actions.
model = joblib.load(model_path)
# Display the app title.
st.title("SuperKart Sales Prediction App")
# Explain the prediction in simple language.
st.write("Enter the product and store details to predict total sales.")
# Collect product weight.
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
# Collect allocated area.
allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.10)
# Collect product MRP.
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
# Collect store establishment year.
store_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2100, value=2009)
# Select product sugar content.
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'No Sugar', 'Regular', 'reg'])
# Select product type.
product_type = st.selectbox("Product Type", ['Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others', 'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'])
# Select store size.
store_size = st.selectbox("Store Size", ['High', 'Medium', 'Small'])
# Select store location city type.
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
# Select store type.
store_type = st.selectbox("Store Type", ['Departmental Store', 'Food Mart', 'Supermarket Type1', 'Supermarket Type2'])
# Build one row using the exact training feature names.
input_data = pd.DataFrame([{"Product_Weight": product_weight, "Product_Allocated_Area": allocated_area, "Product_MRP": product_mrp, "Store_Establishment_Year": store_year, "Product_Sugar_Content": product_sugar_content, "Product_Type": product_type, "Store_Size": store_size, "Store_Location_City_Type": store_location_city_type, "Store_Type": store_type}])
# Run prediction when the button is clicked.
if st.button("Predict Sales"):
    # Predict the sales for the entered details.
    prediction = model.predict(input_data)[0]
    # Display the predicted sales value.
    st.success(f"Predicted Total Sales: {prediction:,.2f}")
