import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# -----------------------------
# Load dataset (Wine Quality)
# NOTE: UCI dataset uses ';' as separator
# -----------------------------
df = pd.read_csv("dataset/winequality.csv", sep=";")

# Clean column names (safety)
df.columns = df.columns.str.strip()

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Preprocessing
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model selection
# -----------------------------
# model = LinearRegression()
# model = Ridge(alpha=1.0)
model = Lasso(alpha=0.01)

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
