import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model (OPTIMIZED to reduce size)
model = RandomForestClassifier(
    n_estimators=50,      # reduce trees
    max_depth=10,         # limit depth
    random_state=42
)

# Train
model.fit(X_scaled, y)

# Save with compression ✅
joblib.dump(model, "fraud_model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)

print("✅ Model saved successfully!")