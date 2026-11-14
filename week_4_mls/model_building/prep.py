# Import pandas to read the dataset.
import pandas as pd
# Import the same splitting function as class.
from sklearn.model_selection import train_test_split
# Read the SuperKart dataset.
df = pd.read_csv("/content/SuperKart (1).csv")
# Remove product and store identifiers.
df.drop(columns=["Product_Id", "Store_Id"], inplace=True)
# Separate input columns from sales.
X = df.drop(columns=["Product_Store_Sales_Total"])
# Select sales as the prediction target.
y = df["Product_Store_Sales_Total"]
# Split the data into 80% training and 20% testing.
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# Save training inputs for the next pipeline job.
Xtrain.to_csv("Xtrain.csv", index=False)
# Save testing inputs for the next pipeline job.
Xtest.to_csv("Xtest.csv", index=False)
# Save training sales for the next pipeline job.
ytrain.to_csv("ytrain.csv", index=False)
# Save testing sales for the next pipeline job.
ytest.to_csv("ytest.csv", index=False)
# Print the sizes of the data splits.
print("Training rows:", len(Xtrain), "Testing rows:", len(Xtest))
