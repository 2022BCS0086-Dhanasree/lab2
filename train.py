import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create output directories if they don't exist
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/winequality.csv")

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing: Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (for GitHub Actions logs)
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save trained model
joblib.dump(model, "output/model/model.pkl")

# Save metrics to JSON
metrics = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
