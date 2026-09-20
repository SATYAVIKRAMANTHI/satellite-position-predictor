import sys
sys.path.insert(0, '.')
from datetime import datetime, timezone

print("=== Testing TLE Fetch (NORAD ID) ===")
from tle_fetcher import fetch_tle
tle = fetch_tle("25544")
print("Name:", tle["name"])
print("NORAD:", tle["norad_id"])
print("Line1:", tle["line1"])
print("Line2:", tle["line2"])
print("From cache:", tle.get("from_cache", False))

print()
print("=== Testing TLE Fetch (Name) ===")
tle2 = fetch_tle("ISS (ZARYA)")
print("Name:", tle2["name"])
print("From cache:", tle2.get("from_cache", False))

print()
print("=== Testing Position Computation ===")
from sgp4_propagator import compute_position
dt = datetime.now(timezone.utc)
pos = compute_position(tle, dt)
print("Latitude :", pos["latitude"], "deg")
print("Longitude:", pos["longitude"], "deg")
print("Altitude :", pos["altitude_km"], "km")
print("Speed    :", pos["speed_km_s"], "km/s")
print("Azimuth  :", pos["azimuth_deg"], "deg")
print("Elevation:", pos["elevation_deg"], "deg")
print("Period   :", pos["orbital_period_min"], "min")
print("Visibility:", pos["visibility"])
print("Track points:", len(pos["ground_track"]["lats"]))

print()
print("=== Testing Feature Engineering ===")
from feature_engineering import parse_tle_elements, generate_training_dataset
elements = parse_tle_elements(tle)
print("Mean motion:", elements["mean_motion"])
print("Inclination:", elements["inclination"])
print("Eccentricity:", elements["eccentricity"])

print()
print("Generating training dataset (24h, 10min steps)...")
df = generate_training_dataset(tle, hours=24, step_minutes=10)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print("Sample lat range:", round(df["latitude"].min(), 2), "to", round(df["latitude"].max(), 2))

print()
print("=== Testing ML Training ===")
from ml_model import train_models, predict_position, is_model_trained
result = train_models(tle, hours=24, step_minutes=10)
print("Best overall model:", result["overall_best"])
print("Total samples:", result["total_samples"])
print("Best per target:", result["best_model"])
print()
for target, bname in result["best_model"].items():
    m = result["metrics"][target][bname]
    print(f"  {target}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")

print()
print("=== Testing Prediction ===")
from datetime import timedelta
future_dt = datetime.now(timezone.utc) + timedelta(hours=6)
pred = predict_position(tle, future_dt)
print("Predicted Latitude :", pred["latitude"])
print("Predicted Longitude:", pred["longitude"])
print("Predicted Altitude :", pred["altitude_km"], "km")
print("Predicted Speed    :", pred["speed_km_s"], "km/s")
print("Model used         :", pred["model_name"])

print()
print("=== All tests PASSED! ===")
