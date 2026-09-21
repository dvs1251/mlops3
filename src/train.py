"""
Ad-hoc Random Forest training script for the Kidney Disease dataset.

Used for the manually-versioned models (V1, V2) before hyperparameters were
moved into params.yaml / dvc.yaml. To reproduce the report's flow:

  1. Leave RF_PARAMS / MODEL_OUT as below, run this file -> Model V1
     (n_estimators=100, max_depth=5)
  2. Edit RF_PARAMS to n_estimators=200, max_depth=10 and MODEL_OUT to
     random_forest_kidney_v2.pkl, run again -> Model V2
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---- change these two between V1 and V2 runs -------------------------
RF_PARAMS = dict(n_estimators=100, max_depth=5, random_state=42)
MODEL_OUT = "models/random_forest_kidney_v1.pkl"
# ------------------------------------------------------------------------

DATA_PATH = "data/cleaned_kidney_disease.csv"
TARGET = "classification"

data = pd.read_csv(DATA_PATH)
print("Dataset shape:", data.shape)

X = data.drop(columns=[TARGET])
y = data[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.str.strip().map({"ckd": 1, "notckd": 0})

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(**RF_PARAMS)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nRANDOM FOREST MODEL")
print("-" * 40)
print(f"Number of trees: {RF_PARAMS['n_estimators']}")
print(f"Maximum depth: {RF_PARAMS['max_depth']}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, MODEL_OUT)
print(f"\nModel saved successfully:\n{MODEL_OUT}")
