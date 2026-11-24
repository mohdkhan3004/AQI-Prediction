"""Train and evaluate the AQI prediction model from the included CSV dataset."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).parent
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]
data = pd.read_csv(ROOT / "data" / "aqi_environmental_data.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data[FEATURES], data["AQI"], test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f}")
print(f"R2  : {r2_score(y_test, pred):.3f}")
joblib.dump(model, ROOT / "aqi_model.joblib")
print("Saved model to aqi_model.joblib")
