"""
DVC pipeline stage: trains the Random Forest kidney-disease model using
hyperparameters from params.yaml, and writes metrics.json.

Invoked by the "train" stage in dvc.yaml (`python src/train_dvc.py`).
"""
import json

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/cleaned_kidney_disease.csv"
TARGET = "classification"
MODEL_OUT = "models/reproducible_kidney_model.pkl"
METRICS_OUT = "metrics.json"

with open("params.yaml") as f:
    params = yaml.safe_load(f)

rf_params = params["random_forest"]
test_size = params["test_size"]

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET])
y = data[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.str.strip().map({"ckd": 1, "notckd": 0})

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        random_state=rf_params["random_state"],
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 5),
    "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 5),
    "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 5),
    "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 5),
}

joblib.dump(pipeline, MODEL_OUT)
with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved to: {MODEL_OUT}")
print(f"Metrics written to: {METRICS_OUT}")
print(metrics)
