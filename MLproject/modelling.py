import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import dagshub

# Replace with your DagsHub username and repository name
REPO_OWNER = "farisgp"  # Replace with your DagsHub username
REPO_NAME = "Eksperimen_SML_FarisGhina"  # Replace with your repository name

# Initialize DagsHub - This helps DagsHub track the experiment correctly
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)


# mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Comment out local tracking URI

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow/")

print("Tracking URI:", mlflow.get_tracking_uri())
# print("DAGSHUB_TOKEN set:", 'DAGSHUB_TOKEN' in os.environ)  # For debugging, can be commented out
# print("Token Value:", os.environ.get("DAGSHUB_TOKEN")[:5], "...(disembunyikan)")  # For debugging, can be commented out

# Create a new MLflow Experiment
mlflow.set_experiment("Clothes Price Prediction")

df = pd.read_csv("clothes_price_prediction_data.csv")

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

with mlflow.start_run():
    mlflow.autolog()
    # Log parameters
    n_estimators = 505
    max_depth = 37
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)