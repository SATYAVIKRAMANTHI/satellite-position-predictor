# 🛰️ Satellite Position Predictor — ML System

A professional, ML-powered satellite position prediction system built with:
- **Streamlit** — interactive web UI
- **SGP4 + Skyfield** — orbital mechanics propagation
- **CelesTrak** — live TLE (Two-Line Element) data
- **Scikit-learn + XGBoost** — ML position forecasting

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔴 Live TLE Fetch | CelesTrak API by name or NORAD ID with 6-hour cache |
| 📡 SGP4 Propagation | Accurate position using SGP4 algorithm + Skyfield |
| 🌍 World Map | Interactive Plotly globe with ground track |
| 🤖 ML Prediction | Random Forest / XGBoost trained on orbital elements |
| 📊 Model Accuracy | MAE, RMSE, R² per target with Actual vs Predicted charts |
| 🔮 Future Prediction | Predict position at any future UTC datetime |
| 📋 TLE Info | Raw TLE display, epoch age warning, cache status |

---

## 📁 Project Structure

```
satellite_predictor/
├── app.py                  # Main Streamlit app (all 5 sections)
├── tle_fetcher.py          # CelesTrak API + file-based cache
├── sgp4_propagator.py      # SGP4 + Skyfield position engine
├── ml_model.py             # ML pipeline: train / predict / evaluate
├── feature_engineering.py  # TLE → ML features + dataset generation
├── utils.py                # Shared helper functions
├── cache/                  # Auto-created TLE cache (JSON files)
│   └── models/             # Trained ML model files
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd satellite_predictor
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## 🔧 Usage Guide

### Section 1 — Search
- Enter a satellite name (e.g., `ISS (ZARYA)`) or a NORAD catalog number (e.g., `25544`)
- Pick a date and time (UTC)
- Click **Compute Position**

### Section 2 — Live Position
- View Latitude, Longitude, Altitude, Speed, Azimuth, Elevation, Orbital Period, Visibility
- See the satellite's position on an interactive world map with ground track

### Section 3 — ML Prediction
1. Click **Train Model** — trains on 7 days of SGP4-propagated data (~2000 samples)
2. Pick a future date/time, click **Predict Position**
3. Compare predicted position with current position on the map

### Section 4 — Model Accuracy
- View MAE, RMSE, R² for each target variable (Lat, Lon, Alt, Speed)
- See Actual vs Predicted charts from the test set
- Use **Retrain Model** to refresh the model with the latest TLE

### Section 5 — Satellite Info
- View satellite name, NORAD ID, TLE epoch date
- See raw TLE lines and parsed orbital elements
- Cache status and TLE age warnings

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Cache TTL | 6 hours | TLE data validity before re-fetching |
| Training hours | 168 h (7 days) | Propagation window for training data |
| Sampling interval | 5 min | Time step between training samples |
| Train/Test split | 80/20 | ML evaluation split |

---

## 🤖 ML Details

**Features used:**
- `epoch_unix` — TLE epoch as Unix timestamp
- `minutes_from_epoch` — Time offset from TLE epoch (primary prediction variable)
- `mean_motion` — Revolutions per day
- `eccentricity` — Orbital eccentricity
- `inclination` — Orbital inclination (degrees)
- `raan` — Right Ascension of Ascending Node (degrees)
- `arg_perigee` — Argument of Perigee (degrees)
- `mean_anomaly` — Mean Anomaly (degrees)
- `bstar` — Drag term
- `rev_number` — Revolution number at epoch

**Targets:**
- Latitude, Longitude, Altitude (km), Speed (km/s)

**Models compared:**
- `Random Forest` (n_estimators=100, max_depth=12)
- `XGBoost` (n_estimators=100, max_depth=6, lr=0.1)

Best model selected per target by R² score on hold-out test set.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web UI |
| sgp4 | ≥2.22 | TLE parsing & propagation |
| skyfield | ≥1.48 | Coordinate transforms |
| scikit-learn | ≥1.4 | Random Forest, scaling, metrics |
| xgboost | ≥2.0 | XGBoost regressor |
| plotly | ≥5.19 | Interactive charts & maps |
| pandas | ≥2.1 | Data manipulation |
| joblib | ≥1.3 | Model serialization |
| requests | ≥2.31 | HTTP API calls |

---

## 🌐 Data Sources

- **TLE Data**: [CelesTrak](https://celestrak.org) — GP Data endpoint
- **Orbital Mechanics**: SGP4 algorithm per AFSPC report (Hoots & Roehrich, 1980)

---

## ⚠️ Limitations

- TLE data accuracy degrades over time (typically < 4 days for LEO)
- ML predictions are trained on SGP4 propagations, not independent observational data
- Azimuth/Elevation computed from the prime meridian (0°N, 0°E) by default
- No authentication required — uses CelesTrak public API

---

*Built with ❤️ using Python, Streamlit, and open orbital mechanics libraries.*
