"""
sgp4_propagator.py
------------------
Propagates a satellite's TLE to a specific UTC datetime using the SGP4
algorithm and Skyfield to compute all orbital parameters.

Outputs:
    - Latitude, Longitude, Altitude (km)
    - Speed (km/s)
    - Azimuth, Elevation (degrees) from a reference ground observer
    - Visibility status
    - Orbital period (minutes)
    - Ground track (list of lat/lon pairs for orbit path)
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    from skyfield.api import load, EarthSatellite, wgs84
    from skyfield.toposlib import GeographicPosition
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False

try:
    from sgp4.api import Satrec, WGS72
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False


# ── Timescale singleton (cached to avoid repeated I/O) ───────────────────────
_TS = None

def _get_timescale():
    global _TS
    if _TS is None:
        _TS = load.timescale()
    return _TS


# ── Main propagation function ─────────────────────────────────────────────────

def compute_position(
    tle_data: dict,
    target_dt: datetime,
    observer_lat: float = 0.0,
    observer_lon: float = 0.0,
    observer_elev_m: float = 0.0,
) -> dict:
    """
    Compute satellite position at a given UTC datetime.

    Parameters
    ----------
    tle_data     : dict   — TLE dict with keys: name, line1, line2
    target_dt    : datetime — Target UTC datetime (must be timezone-aware)
    observer_lat : float  — Observer latitude in degrees (default: 0°N)
    observer_lon : float  — Observer longitude in degrees (default: 0°E)
    observer_elev_m : float — Observer elevation in metres (default: 0 m)

    Returns
    -------
    dict with:
        latitude, longitude, altitude_km, speed_km_s,
        azimuth_deg, elevation_deg, visibility,
        orbital_period_min, ground_track
    """
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError(
            "Skyfield is not installed. Run: pip install skyfield"
        )

    name = tle_data.get("name", "UNKNOWN")
    line1 = tle_data["line1"]
    line2 = tle_data["line2"]

    # Ensure UTC
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    else:
        target_dt = target_dt.astimezone(timezone.utc)

    ts = _get_timescale()
    satellite = EarthSatellite(line1, line2, name, ts)

    # Build Skyfield time object
    t = ts.from_datetime(target_dt)

    # ── Subpoint (lat, lon, alt) ─────────────────────────────────────────────
    geocentric = satellite.at(t)

    # Check if satellite is deep-space (e.g. JWST at L2 ~1.5M km)
    # wgs84.geographic_position_of() only works reliably for Earth-orbit satellites
    try:
        pos_km = geocentric.position.km
        dist_from_earth_km = math.sqrt(float(pos_km[0])**2 + float(pos_km[1])**2 + float(pos_km[2])**2)
        is_deep_space = dist_from_earth_km > 100_000  # beyond 100,000 km = deep space
    except Exception:
        dist_from_earth_km = 0.0
        is_deep_space = False

    if is_deep_space:
        # For deep-space objects, compute lat/lon from GCRS position vector directly
        x, y, z = float(pos_km[0]), float(pos_km[1]), float(pos_km[2])
        lat_deg = math.degrees(math.asin(z / dist_from_earth_km))
        lon_deg = math.degrees(math.atan2(y, x))
        alt_km = dist_from_earth_km - 6371.0  # subtract Earth radius
    else:
        subpoint = wgs84.geographic_position_of(geocentric)
        lat_deg = subpoint.latitude.degrees
        lon_deg = subpoint.longitude.degrees
        alt_km = subpoint.elevation.km

    # ── Velocity / Speed ─────────────────────────────────────────────────────
    try:
        vel_array = geocentric.velocity.km_per_s  # shape (3,)
        speed_km_s = float(math.sqrt(float(vel_array[0])**2 +
                                     float(vel_array[1])**2 +
                                     float(vel_array[2])**2))
    except Exception:
        # Fallback using sgp4 directly
        speed_km_s = _compute_speed_sgp4(line1, line2, target_dt)

    # ── Azimuth & Elevation from observer ────────────────────────────────────
    try:
        observer = wgs84.latlon(observer_lat, observer_lon, elevation_m=observer_elev_m)
        difference = satellite - observer
        topocentric = difference.at(t)
        alt_angle, az_angle, _ = topocentric.altaz()
        elevation_deg = float(alt_angle.degrees)
        azimuth_deg = float(az_angle.degrees)
    except Exception:
        elevation_deg = -90.0
        azimuth_deg = 0.0

    # ── Visibility ───────────────────────────────────────────────────────────
    if is_deep_space:
        visibility = "Deep Space (not in Earth orbit)"
    else:
        visibility = "Visible" if elevation_deg > 0 else "Below Horizon"

    # ── Orbital period ───────────────────────────────────────────────────────
    orbital_period_min = _compute_orbital_period(line2)

    # ── Ground track ─────────────────────────────────────────────────────────
    if is_deep_space:
        ground_track = {"lats": [round(lat_deg, 4)], "lons": [round(lon_deg, 4)]}
    else:
        ground_track = _compute_ground_track(satellite, ts, target_dt, orbital_period_min)

    return {
        "name": name,
        "latitude": round(lat_deg, 4),
        "longitude": round(lon_deg, 4),
        "altitude_km": round(alt_km, 3),
        "speed_km_s": round(speed_km_s, 4),
        "azimuth_deg": round(azimuth_deg, 2),
        "elevation_deg": round(elevation_deg, 2),
        "visibility": visibility,
        "orbital_period_min": round(orbital_period_min, 2),
        "ground_track": ground_track,
        "observer_lat": observer_lat,
        "observer_lon": observer_lon,
    }


def _compute_speed_sgp4(line1: str, line2: str, target_dt: datetime) -> float:
    """Fallback speed computation using sgp4 library directly."""
    if not SGP4_AVAILABLE:
        return 0.0
    try:
        satellite = Satrec.twoline2rv(line1, line2)
        jd, fr = _datetime_to_jd(target_dt)
        e, r, v = satellite.sgp4(jd, fr)
        if e != 0:
            return 0.0
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    except Exception:
        return 0.0


def _datetime_to_jd(dt: datetime):
    """Convert datetime to Julian Date (jd, fraction)."""
    # Julian date for J2000.0 epoch (2000-01-01 12:00 TT)
    # Simple conversion
    import time as _time
    import calendar
    gmtime = calendar.timegm(dt.timetuple())
    jd = 2440587.5 + gmtime / 86400.0
    # Split into integer and fraction
    jd_int = math.floor(jd)
    jd_frac = jd - jd_int
    return float(jd_int), float(jd_frac)


def _compute_orbital_period(line2: str) -> float:
    """
    Compute orbital period from TLE Line 2.
    Period (min) = 1440 / mean_motion (rev/day)
    """
    try:
        mean_motion_rev_day = float(line2[52:63].strip())
        if mean_motion_rev_day <= 0:
            return 0.0
        return 1440.0 / mean_motion_rev_day
    except (ValueError, IndexError):
        return 92.0  # Default ISS-like period


def _compute_ground_track(satellite, ts, center_dt: datetime, period_min: float) -> dict:
    """
    Compute the satellite ground track (one full orbit ± half period).

    Returns
    -------
    dict: {lats: [...], lons: [...]}
    """
    if period_min <= 0:
        period_min = 92.0

    step_min = max(1.0, period_min / 90)  # ~90 points per orbit
    half_period = period_min / 2.0

    lats, lons = [], []
    t_start = center_dt - timedelta(minutes=half_period)

    num_steps = int(period_min / step_min) + 1
    for i in range(num_steps):
        t_offset = t_start + timedelta(minutes=i * step_min)
        try:
            t_sky = ts.from_datetime(t_offset.replace(tzinfo=timezone.utc)
                                     if t_offset.tzinfo is None
                                     else t_offset.astimezone(timezone.utc))
            geo = satellite.at(t_sky)
            sp = wgs84.geographic_position_of(geo)
            lats.append(round(sp.latitude.degrees, 3))
            lons.append(round(sp.longitude.degrees, 3))
        except Exception:
            continue

    return {"lats": lats, "lons": lons}


def get_tle_epoch_datetime(tle_data: dict) -> Optional[datetime]:
    """
    Extract the TLE epoch as a UTC datetime from Line 1.

    TLE Line 1 columns 19-32: epoch in YYDDD.DDDDDDDD format
    """
    try:
        line1 = tle_data["line1"]
        epoch_str = line1[18:32].strip()
        year_2digit = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])

        # Convert 2-digit year to 4-digit
        year = 2000 + year_2digit if year_2digit < 57 else 1900 + year_2digit

        # Convert day of year to datetime
        epoch_dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
        return epoch_dt
    except (ValueError, IndexError, KeyError):
        return None
