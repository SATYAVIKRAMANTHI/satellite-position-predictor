"""
feature_engineering.py
-----------------------
Converts TLE data into ML features by propagating the satellite
over a time range (epoch → epoch + N hours) at fixed intervals.

Each row of the resulting DataFrame represents one point in time and contains:
    FEATURES (X):
        epoch_unix, mean_motion, eccentricity, inclination, raan,
        arg_perigee, mean_anomaly, bstar, rev_number,
        minutes_from_epoch

    TARGETS (y):
        latitude, longitude, altitude_km, speed_km_s
"""

import math
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Skyfield / sgp4 (imported lazily to avoid hard crash at module load)
try:
    from skyfield.api import load, EarthSatellite, wgs84
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

try:
    from sgp4.api import Satrec
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False


# ── TLE element parsers ───────────────────────────────────────────────────────

def parse_tle_elements(tle_data: dict) -> dict:
    """
    Parse orbital elements from TLE Line 1 and Line 2.

    Returns
    -------
    dict with keys:
        epoch_unix, mean_motion, eccentricity, inclination,
        raan, arg_perigee, mean_anomaly, bstar, rev_number, epoch_dt
    """
    line1 = tle_data["line1"]
    line2 = tle_data["line2"]

    # ── Line 1 elements ──────────────────────────────────────────────────
    # Epoch
    epoch_str = line1[18:32].strip()
    epoch_dt = _parse_tle_epoch(epoch_str)
    epoch_unix = epoch_dt.timestamp() if epoch_dt else 0.0

    # BSTAR drag term (mantissa × 10^exponent)
    bstar = _parse_bstar(line1[53:61].strip(), line1[61:63].strip())

    # Revolution number at epoch
    try:
        rev_number = int(line1[63:68].strip())
    except (ValueError, IndexError):
        rev_number = 0

    # ── Line 2 elements ──────────────────────────────────────────────────
    try:
        inclination = float(line2[8:16].strip())
    except (ValueError, IndexError):
        inclination = 0.0

    try:
        raan = float(line2[17:25].strip())
    except (ValueError, IndexError):
        raan = 0.0

    try:
        # Eccentricity: stored without decimal point (e.g. 0007416 → 0.0007416)
        eccentricity = float("0." + line2[26:33].strip())
    except (ValueError, IndexError):
        eccentricity = 0.0

    try:
        arg_perigee = float(line2[34:42].strip())
    except (ValueError, IndexError):
        arg_perigee = 0.0

    try:
        mean_anomaly = float(line2[43:51].strip())
    except (ValueError, IndexError):
        mean_anomaly = 0.0

    try:
        mean_motion = float(line2[52:63].strip())
    except (ValueError, IndexError):
        mean_motion = 15.5  # default ISS-like

    return {
        "epoch_unix": epoch_unix,
        "epoch_dt": epoch_dt,
        "mean_motion": mean_motion,
        "eccentricity": eccentricity,
        "inclination": inclination,
        "raan": raan,
        "arg_perigee": arg_perigee,
        "mean_anomaly": mean_anomaly,
        "bstar": bstar,
        "rev_number": rev_number,
    }


def _parse_tle_epoch(epoch_str: str) -> Optional[datetime]:
    """Convert TLE epoch string (YYDDD.DDDDDDDD) to UTC datetime."""
    try:
        year_2d = int(epoch_str[:2])
        day_frac = float(epoch_str[2:])
        year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_frac - 1)
        return dt
    except (ValueError, IndexError):
        return datetime.now(timezone.utc)


def _parse_bstar(mantissa_str: str, exp_str: str) -> float:
    """
    Parse the BSTAR drag term from TLE Line 1.
    Format: ±.MMMMMMM±EE  (implied decimal before mantissa)
    """
    try:
        # Handle formats like " 00000-0" or " 12345-3"
        full = (mantissa_str + exp_str).replace(" ", "")
        if not full:
            return 0.0
        # Find sign and exponent
        for i in range(len(full) - 1, 0, -1):
            if full[i] in ("+", "-") and i > 0:
                mantissa = float(full[:i]) * (1e-5 if len(full[:i]) >= 5 else 1.0)
                exp = int(full[i:])
                return mantissa * (10 ** exp)
        return float(full) if full else 0.0
    except (ValueError, IndexError):
        return 0.0


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_training_dataset(
    tle_data: dict,
    hours: int = 168,
    step_minutes: int = 5,
) -> pd.DataFrame:
    """
    Generate a training dataset by propagating the satellite forward from
    its TLE epoch in fixed time steps.

    Parameters
    ----------
    tle_data     : dict  — TLE as returned by tle_fetcher
    hours        : int   — Duration to propagate (default: 7 days = 168 h)
    step_minutes : int   — Time step in minutes (default: 5)

    Returns
    -------
    pd.DataFrame with feature columns + target columns
    """
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError("Skyfield is required. Run: pip install skyfield")

    elements = parse_tle_elements(tle_data)
    epoch_dt = elements["epoch_dt"]
    if epoch_dt is None:
        epoch_dt = datetime.now(timezone.utc)

    ts = load.timescale()
    name = tle_data.get("name", "SAT")
    satellite = EarthSatellite(tle_data["line1"], tle_data["line2"], name, ts)

    records = []
    total_minutes = hours * 60
    num_steps = total_minutes // step_minutes

    for i in range(num_steps):
        minutes_offset = i * step_minutes
        t_dt = epoch_dt + timedelta(minutes=minutes_offset)

        try:
            t_sky = ts.from_datetime(t_dt)
            geocentric = satellite.at(t_sky)
            # geographic_position_of returns correct elevation; subpoint_of clips to 0
            subpoint = wgs84.geographic_position_of(geocentric)

            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km

            vel = geocentric.velocity.km_per_s
            speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

            records.append({
                # Features
                "epoch_unix": elements["epoch_unix"],
                "minutes_from_epoch": float(minutes_offset),
                "mean_motion": elements["mean_motion"],
                "eccentricity": elements["eccentricity"],
                "inclination": elements["inclination"],
                "raan": elements["raan"],
                "arg_perigee": elements["arg_perigee"],
                "mean_anomaly": elements["mean_anomaly"],
                "bstar": elements["bstar"],
                "rev_number": float(elements["rev_number"]),
                # Targets
                "latitude": lat,
                "longitude": lon,
                "altitude_km": alt,
                "speed_km_s": speed,
            })
        except Exception:
            continue

    if len(records) == 0:
        raise ValueError(
            "Could not generate any training data. "
            "Check that the TLE is valid and not too old."
        )

    df = pd.DataFrame(records)
    return df


# Column definitions
FEATURE_COLS = [
    "epoch_unix",
    "minutes_from_epoch",
    "mean_motion",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
    "mean_anomaly",
    "bstar",
    "rev_number",
]

TARGET_COLS = ["latitude", "longitude", "altitude_km", "speed_km_s"]


def build_prediction_features(
    tle_data: dict, target_dt: datetime
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for ML prediction at a given datetime.

    Parameters
    ----------
    tle_data  : dict     — TLE dict
    target_dt : datetime — UTC datetime to predict for

    Returns
    -------
    pd.DataFrame with one row containing the feature columns
    """
    elements = parse_tle_elements(tle_data)
    epoch_dt = elements["epoch_dt"]

    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    if epoch_dt.tzinfo is None:
        epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)

    minutes_from_epoch = (target_dt - epoch_dt).total_seconds() / 60.0

    row = {
        "epoch_unix": elements["epoch_unix"],
        "minutes_from_epoch": minutes_from_epoch,
        "mean_motion": elements["mean_motion"],
        "eccentricity": elements["eccentricity"],
        "inclination": elements["inclination"],
        "raan": elements["raan"],
        "arg_perigee": elements["arg_perigee"],
        "mean_anomaly": elements["mean_anomaly"],
        "bstar": elements["bstar"],
        "rev_number": float(elements["rev_number"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)
