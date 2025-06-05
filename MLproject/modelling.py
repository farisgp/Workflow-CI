import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import joblib
import dagshub
import argparse

REPO_OWNER = "farisgp"  
REPO_NAME = "Eksperimen_SML_FarisGhina"  

try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    print("Dagshub initialization successful.")
except Exception as e:
    print(f"Error initializing Dagshub: {e}")
    raise


# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")

print("Tracking URI:", mlflow.get_tracking_uri())
# print("DAGSHUB_TOKEN set:", 'DAGSHUB_TOKEN' in os.environ)  # For debugging, can be commented out
# print("Token Value:", os.environ.get("DAGSHUB_TOKEN")[:5], "...(disembunyikan)")  # For debugging, can be commented out
# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/farisgp/Eksperimen_SML_FarisGhina.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "farisgp"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "0df17c107d7137b8f9a7fe14bb6c6057d4d68db5")
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://dagshub.com/farisgp/Eksperimen_SML_FarisGhina.s3"
# os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MLFLOW_TRACKING_PUBLIC_KEY", "0df17c107d7137b8f9a7fe14bb6c6057d4d68db5 ")
# os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "0df17c107d7137b8f9a7fe14bb6c6057d4d68db5")
# os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# print("MLFLOW_TRACKING_PASSWORD:", os.getenv("MLFLOW_TRACKING_PASSWORD", "Not set"))
# print("AWS_SECRET_ACCESS_KEY:", os.getenv("MLFLOW_TRACKING_PASSWORD", "Not set"))


# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price - CI")

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
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())

    # Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")


    # joblib.dump(model, "model.pkl")
    joblib.dump(model, "model.pkl")
    print(f"Model saved as 'model.pkl'")

# Ambil run yang sedang aktif dari MLflow (karena mlflow run sudah memulai run)
# run = mlflow.active_run()
# if run is None:
#     raise RuntimeError("No active MLflow run found. This script must be run using `mlflow run`.")

# mlflow.autolog()  # Autolog model, params, metrics

# # Training
# model = RandomForestRegressor(
#     n_estimators=args.n_estimators,
#     max_depth=args.max_depth,
#     random_state=42
# )
# model.fit(X_train, y_train)

# # Prediksi dan evaluasi
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Logging manual (optional karena autolog juga sudah menangkap)
# mlflow.log_param("n_estimators", args.n_estimators)
# mlflow.log_param("max_depth", args.max_depth)
# mlflow.log_metric("MSE", mse)
# mlflow.log_metric("R2", r2)

# # Logging model
# mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())

# # Simpan model ke file
# joblib.dump(model, "model.pkl")
# print("Model saved to model.pkl")

# # Simpan run_id ke file agar bisa dibaca GitHub Actions
# run_id = run.info.run_id
# with open("run_id.txt", "w") as f:
#     f.write(run_id)
# print(f"Run ID saved to run_id.txt: {run_id}")
