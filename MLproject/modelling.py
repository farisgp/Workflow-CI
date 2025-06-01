import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy as np
import os
import joblib
import dagshub
import argparse

REPO_OWNER = "farisgp"  
REPO_NAME = "Eksperimen_SML_FarisGhina"  

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")

print("Tracking URI:", mlflow.get_tracking_uri())
# print("DAGSHUB_TOKEN set:", 'DAGSHUB_TOKEN' in os.environ)  # For debugging, can be commented out
# print("Token Value:", os.environ.get("DAGSHUB_TOKEN")[:5], "...(disembunyikan)")  # For debugging, can be commented out

# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price Prediction")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="clothes_price_prediction_preprocessing.csv")
args = parser.parse_args()

df = pd.read_csv(args.data_path)

# Encode fitur kategorikal
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

X = df.drop("Price", axis=1)
y = df["Price"]

# Normalisasi Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

input_example = X_train[0:5]

with mlflow.start_run() as run:
    mlflow.autolog()
    # Log parameters
    n_estimators = 505
    max_depth = 37
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("clasification report:\n", report)

    # Cetak run_id agar bisa digunakan di GitHub Actions
    run_id = run.info.run_id
    print(f"MLFLOW_RUN_ID={run_id}")

    joblib.dump(model, os.makedirs(os.path.dirname("./models/model.pkl"), exist_ok=True))
    print(f"Model saved to: ./models/model.pkl")
