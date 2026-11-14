# Import pandas for reading saved train/test files.
import pandas as pd
# Import scaling and encoding methods from class.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Import the class preprocessing function.
from sklearn.compose import make_column_transformer
# Import the class pipeline function.
from sklearn.pipeline import make_pipeline
# Import XGBoost for sales regression.
import xgboost as xgb
# Import GridSearchCV to tune the model.
from sklearn.model_selection import GridSearchCV
# Import regression evaluation methods.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Import joblib to save the trained model.
import joblib
# Import MLflow to record the experiment.
import mlflow
# Connect to the MLflow server started by GitHub Actions.
mlflow.set_tracking_uri("http://localhost:5000")
# Name the production experiment.
mlflow.set_experiment("superkart-training-experiment")
# Load the training inputs saved by preparation.
Xtrain = pd.read_csv("Xtrain.csv")
# Load the testing inputs saved by preparation.
Xtest = pd.read_csv("Xtest.csv")
# Load training sales as a Series.
ytrain = pd.read_csv("ytrain.csv").squeeze()
# Load testing sales as a Series.
ytest = pd.read_csv("ytest.csv").squeeze()
# List numerical input columns.
numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
# List categorical input columns.
categorical_features = ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]
# Scale numerical values and one-hot encode categories.
preprocessor = make_column_transformer((StandardScaler(), numeric_features), (OneHotEncoder(handle_unknown="ignore"), categorical_features))
# Create the XGBoost regression model.
xgb_model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")
# Define a small grid for GitHub Actions.
param_grid = {"xgbregressor__n_estimators": [50, 100], "xgbregressor__max_depth": [2, 3], "xgbregressor__learning_rate": [0.05, 0.1]}
# Combine preprocessing and training as in class.
model_pipeline = make_pipeline(preprocessor, xgb_model)
# Start the main MLflow run.
with mlflow.start_run():
    # Tune the model using five-fold cross-validation.
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    # Train and tune using the training data.
    grid_search.fit(Xtrain, ytrain)
    # Collect the tuning results.
    results = grid_search.cv_results_
    # Log each tested combination in MLflow.
    for i in range(len(results["params"])):
        # Start a nested run.
        with mlflow.start_run(nested=True):
            # Log the parameters tested.
            mlflow.log_params(results["params"][i])
            # Log the average cross-validation score.
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
    # Log the best parameters in the main run.
    mlflow.log_params(grid_search.best_params_)
    # Select the best trained model.
    best_model = grid_search.best_estimator_
    # Predict the training sales.
    train_pred = best_model.predict(Xtrain)
    # Predict the testing sales.
    test_pred = best_model.predict(Xtest)
    # Calculate regression metrics.
    metrics = {"train_mae": mean_absolute_error(ytrain, train_pred), "test_mae": mean_absolute_error(ytest, test_pred), "train_rmse": mean_squared_error(ytrain, train_pred) ** 0.5, "test_rmse": mean_squared_error(ytest, test_pred) ** 0.5, "train_r2": r2_score(ytrain, train_pred), "test_r2": r2_score(ytest, test_pred)}
    # Log the actual metrics to MLflow.
    mlflow.log_metrics(metrics)
    # Show the best parameters.
    print("Best parameters:", grid_search.best_params_)
    # Show the model's actual performance.
    print("Model metrics:", metrics)
    # Set the model location for Streamlit.
    model_path = "week_3_mls/deployment/best_superkart_model_v1.joblib"
    # Save the trained model to a file.
    joblib.dump(best_model, model_path)
    # Save the model as an MLflow artifact.
    mlflow.log_artifact(model_path, artifact_path="model")
    # Confirm that the model was saved.
    print("Model saved to", model_path)
