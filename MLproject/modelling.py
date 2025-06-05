import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, r2_score
import random
import numpy as np
import os
import joblib
import dagshub
import argparse

REPO_OWNER = "farisgp"  
REPO_NAME = "Eksperimen_SML_FarisGhina"  

# dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")

print("Tracking URI:", mlflow.get_tracking_uri())
# print("DAGSHUB_TOKEN set:", 'DAGSHUB_TOKEN' in os.environ)  # For debugging, can be commented out
# print("Token Value:", os.environ.get("DAGSHUB_TOKEN")[:5], "...(disembunyikan)")  # For debugging, can be commented out

# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price - CI")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

n_estimators = args.n_estimators
max_depth = args.max_depth

# --- Load Preprocessed Data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # pastikan jadi 1D
y_test = pd.read_csv("y_test.csv").values.ravel()  # pastikan jadi 1D

with mlflow.start_run() as run:
    mlflow.autolog()
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logging parameter dan metrik secara manual
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())

    # Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")

    # Simpan run_id ke file agar bisa diambil GitHub Actions
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    # joblib.dump(model, "model.pkl")
    joblib.dump(model, "model.pkl")
    print(f"Model saved as 'model.pkl'")

# Dapatkan run MLflow yang sudah aktif
# active_run = mlflow.active_run()

# if active_run is None:
#     print("WARNING: No active MLflow run found. This script should be run via 'mlflow run MLproject'.")
#     run_id = "NO_ACTIVE_RUN_ID"
# else:
#     # Semua operasi logging (log_param, log_metric, log_model)
#     # akan otomatis terkait dengan active_run ini.

#     model = RandomForestRegressor(
#         n_estimators=args.n_estimators,
#         max_depth=args.max_depth,
#         random_state=42
#     )
#     # y_train sudah di-ravel() di atas, jadi tidak perlu .values.ravel() lagi di sini
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Logging parameter dan metrik secara otomatis terhubung ke active_run
#     mlflow.log_param("n_estimators", args.n_estimators)
#     mlflow.log_param("max_depth", args.max_depth)
#     mlflow.log_metric("MSE", mse)
#     mlflow.log_metric("R2", r2)

#     # Log model
#     mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head())

#     # Dapatkan run_id dari active_run
#     run_id = active_run.info.run_id

#     # Simpan model ke file PKL di direktori yang sama dengan script modelling.py
#     joblib.dump(model, os.path.join(os.path.dirname(__file__), "model.pkl"))
#     print(f"Model saved as '{os.path.join(os.path.dirname(__file__), 'model.pkl')}'")

# # Cetak run_id agar bisa diambil oleh GitHub Actions
# # Baris ini harus selalu dieksekusi setelah logika modelling,
# # terlepas dari apakah active_run ditemukan atau tidak (untuk debugging)
# print(f"MLFLOW_RUN_ID={run_id}", flush=True)
