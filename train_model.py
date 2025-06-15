import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load and clean the dataset
df = pd.read_csv("male_players.csv", encoding="utf-8", on_bad_lines="skip", engine="python")

# Select features and target
features = ["age", "overall", "potential", "wage_eur"]
target = "value_eur"

# Drop rows with missing values
df = df[features + [target]].dropna()

# Split features and target
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Output results
print("✅ Model Trained: RandomForestRegressor")
print("R² Score:", round(r2, 4))
print("RMSE:", round(rmse, 4))

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_columns.pkl")
print("✅ model.pkl, scaler.pkl, feature_columns.pkl saved.")

# Save prediction log
log_df = pd.DataFrame({
    "actual": y_test.reset_index(drop=True),
    "predicted": y_pred
})
log_df.to_csv("model_results.csv", index=False)
print("📝 model_results.csv saved with prediction logs.")
