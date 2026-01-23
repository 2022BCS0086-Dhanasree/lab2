import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset/winequality.csv", sep=";")
df.columns = df.columns.str.strip()

# -----------------------------
# SELECT EXACT FEATURES USED IN app.py
# -----------------------------
FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "pH",
    "sulphates",
    "alcohol"
]

X = df[FEATURE_COLUMNS]
y = df["quality"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    random_state=7,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, "output/model/model.pkl")

# -----------------------------
# Save metrics
# -----------------------------
metrics = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
