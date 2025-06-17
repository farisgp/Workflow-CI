import mlflow
import dagshub
import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
import joblib

REPO_OWNER = "farisgp"  
REPO_NAME = "Eksperimen_SML_FarisGhina"  
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")
mlflow.set_experiment("Clothes Price CI")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# --- Load Preprocessed Data
df = pd.read_csv(args.data_path)

# Pisahkan fitur dan target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logging parameter dan metrik secara manual
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Log model
    # mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())
    # Deteksi apakah sedang berjalan di GitHub Actions
    if os.getenv("GITHUB_ACTIONS") is None:
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")

    # Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")


    # joblib.dump(model, "model.pkl")
    joblib.dump(model, "model.pkl")
    print(f"Model saved as 'model.pkl'")
