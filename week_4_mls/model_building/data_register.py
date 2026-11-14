# Import pandas to read the sales data.
import pandas as pd
# Set the location of the uploaded dataset.
RAW_PATH = "/content/SuperKart (1).csv"
# Read the SuperKart dataset.
df = pd.read_csv(RAW_PATH)
# List the columns expected in the dataset.
expected_columns = ["Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total"]
# Find any missing columns.
missing = [c for c in expected_columns if c not in df.columns]
# Stop if any required columns are missing.
if missing:
    raise ValueError(f"Missing columns: {missing}")
# Confirm that the dataset was loaded.
print("Dataset registered successfully.")
# Show the dataset size.
print("Rows:", df.shape[0], "Columns:", df.shape[1])
# Show the number of missing values.
print("Missing values:", df.isnull().sum().sum())
# Show the average sales value.
print("Average sales:", round(df["Product_Store_Sales_Total"].mean(), 2))
