"""
tle_fetcher.py
--------------
Fetches Two-Line Element (TLE) data from CelesTrak's GP data API.

Working endpoints (verified 2026):
  By NORAD ID (TLE text):
    https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE
  By NORAD ID (JSON with orbital elements):
    https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=JSON
  By group (TLE text):
    https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=TLE
  Static TLE files (large catalog):
    https://celestrak.org/NORAD/elements/stations.txt
    https://celestrak.org/NORAD/elements/visual.txt

Cache: TLE data is stored in cache/<norad_id>.json for 6 hours.
"""

import os
import json
import time
import warnings
import requests
from datetime import datetime, timezone

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_TTL_SECONDS = 6 * 3600   # 6 hours

# Verified working CelesTrak GP endpoint (2024+)
GP_BASE = "https://celestrak.org/NORAD/elements/gp.php"

# Group-based static TLE files for broad name searches
GROUP_URLS = [
    "https://celestrak.org/NORAD/elements/stations.txt",
    "https://celestrak.org/NORAD/elements/visual.txt",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=resource&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=noaa&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=goes&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium&FORMAT=TLE",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=TLE",
]

HEADERS = {
    "User-Agent": "SatellitePredictor/1.0 (+https://github.com/educational)"
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(key: str) -> str:
    safe = str(key).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _read_cache(key: str):
    """Return cached data if it exists and is fresh, else None."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("cached_at", 0) < CACHE_TTL_SECONDS:
            data["from_cache"] = True
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _write_cache(key: str, data: dict):
    _ensure_cache_dir()
    data = dict(data)
    data["cached_at"] = time.time()
    data["from_cache"] = False
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[Cache] Warning: {e}")


# ── TLE text parser ───────────────────────────────────────────────────────────

def _parse_tle_block(tle_text: str) -> dict:
    """
    Parse the first valid TLE triplet from a block of TLE text.
    Returns dict with: name, line1, line2, norad_id
    Raises ValueError if no valid TLE found.
    """
    lines = [ln.rstrip() for ln in tle_text.strip().splitlines() if ln.strip()]
    i = 0
    while i < len(lines) - 2:
        l1 = lines[i + 1] if i + 1 < len(lines) else ""
        l2 = lines[i + 2] if i + 2 < len(lines) else ""
        if l1.startswith("1 ") and l2.startswith("2 ") and len(l1) >= 69 and len(l2) >= 69:
            name = lines[i].strip()
            try:
                norad_id = int(l1[2:7].strip())
            except ValueError:
                norad_id = None
            return {
                "name": name,
                "line1": l1,
                "line2": l2,
                "norad_id": norad_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        i += 1
    raise ValueError("No valid TLE triplet found in response")


def _search_tle_text_for_name(tle_text: str, name_query: str) -> dict:
    """
    Search a multi-TLE text block for a specific satellite name.
    Returns dict with: name, line1, line2, norad_id
    """
    name_upper = name_query.strip().upper()
    lines = [ln.rstrip() for ln in tle_text.strip().splitlines() if ln.strip()]

    for i in range(len(lines) - 2):
        sat_name_line = lines[i].strip().upper()
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        # Check if name matches AND lines look like valid TLE
        if (name_upper in sat_name_line and
                l1.startswith("1 ") and l2.startswith("2 ") and
                len(l1) >= 69 and len(l2) >= 69):
            try:
                norad_id = int(l1[2:7].strip())
            except ValueError:
                norad_id = None
            return {
                "name": lines[i].strip(),
                "line1": l1,
                "line2": l2,
                "norad_id": norad_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

    raise ValueError(f"Satellite '{name_query}' not found in TLE catalog")


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 20) -> requests.Response:
    """Make a GET request with retries, suppressing SSL warnings."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    raise last_exc


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_tle_by_norad(norad_id: int) -> dict:
    """
    Fetch TLE for a satellite by NORAD catalog number.

    Parameters
    ----------
    norad_id : int   (e.g. 25544 for ISS)

    Returns
    -------
    dict: {name, line1, line2, norad_id, fetched_at, from_cache}
    """
    cache_key = f"norad_{norad_id}"
    cached = _read_cache(cache_key)
    if cached:
        return cached

    errors = []

    # ── Attempt 1: TLE text format ──────────────────────────────────────────
    url_tle = f"{GP_BASE}?CATNR={norad_id}&FORMAT=TLE"
    try:
        resp = _get(url_tle)
        resp.raise_for_status()
        text = resp.text.strip()
        if text and len(text) > 10 and "<html" not in text.lower():
            result = _parse_tle_block(text)
            _write_cache(cache_key, result)
            if result.get("name"):
                _write_cache(f"name_{result['name'].upper().replace(' ','_')}", dict(result))
            return result
        errors.append(f"TLE endpoint returned empty/invalid response")
    except Exception as e:
        errors.append(f"TLE endpoint: {e}")

    # ── Attempt 2: JSON format fallback ─────────────────────────────────────
    url_json = f"{GP_BASE}?CATNR={norad_id}&FORMAT=JSON"
    try:
        resp = _get(url_json)
        resp.raise_for_status()
        data_list = resp.json()
        if isinstance(data_list, list) and len(data_list) > 0:
            gp = data_list[0]
            # Build TLE lines from JSON GP elements
            name = gp.get("OBJECT_NAME", f"NORAD-{norad_id}")
            line1 = gp.get("TLE_LINE1", "")
            line2 = gp.get("TLE_LINE2", "")
            if line1 and line2:
                result = {
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "norad_id": norad_id,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_cache(cache_key, result)
                if name:
                    _write_cache(f"name_{name.upper().replace(' ','_')}", dict(result))
                return result
        errors.append("JSON endpoint returned no data")
    except Exception as e:
        errors.append(f"JSON endpoint: {e}")

    raise ValueError(
        f"Could not fetch TLE for NORAD ID {norad_id}.\n"
        f"Details: {'; '.join(errors)}\n"
        "Tips:\n"
        "  • Check your internet connection\n"
        "  • Try a different satellite catalog number\n"
        f"  • Verify the number at: https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    )


def fetch_tle_by_name(satellite_name: str) -> dict:
    """
    Fetch TLE for a satellite by name.

    Strategy:
    1. Try direct CelesTrak GP NAME search (matches name fragment)
    2. If no result, search through group TLE file catalogs

    Parameters
    ----------
    satellite_name : str  (e.g. "ISS (ZARYA)", "HUBBLE", "NOAA-19")

    Returns
    -------
    dict: {name, line1, line2, norad_id, fetched_at, from_cache}
    """
    name_clean = satellite_name.strip()
    cache_key = f"name_{name_clean.upper().replace(' ','_').replace('(','').replace(')','')}"
    cached = _read_cache(cache_key)
    if cached:
        return cached

    # Step 1: Direct GP name search (returns all matches with name fragment)
    url = f"{GP_BASE}?NAME={requests.utils.quote(name_clean)}&FORMAT=TLE"
    try:
        resp = _get(url)
        resp.raise_for_status()
        text = resp.text.strip()
        if text and len(text) > 10 and not ("<html" in text.lower()):
            # Try exact match first, fallback to first result
            try:
                result = _search_tle_text_for_name(text, name_clean)
            except ValueError:
                result = _parse_tle_block(text)  # take the first result

            _write_cache(cache_key, result)
            _write_cache(f"norad_{result['norad_id']}", dict(result))
            return result
    except (requests.RequestException, ValueError):
        pass  # Fall through to group search

    # Step 2: Search through group catalogs
    name_upper = name_clean.upper()
    for group_url in GROUP_URLS:
        try:
            resp = _get(group_url, timeout=20)
            if resp.status_code != 200:
                continue
            text = resp.text.strip()
            if not text or "<html" in text.lower():
                continue
            try:
                result = _search_tle_text_for_name(text, name_clean)
                _write_cache(cache_key, result)
                _write_cache(f"norad_{result['norad_id']}", dict(result))
                return result
            except ValueError:
                continue
        except requests.RequestException:
            continue

    raise ValueError(
        f"Satellite '{satellite_name}' not found.\n"
        "Tips:\n"
        "  • Try the NORAD catalog number instead (e.g. 25544 for ISS)\n"
        "  • Use uppercase name as listed on CelesTrak (e.g. 'ISS (ZARYA)')\n"
        "  • Try a shorter name fragment (e.g. 'HUBBLE' instead of 'Hubble Space Telescope')"
    )


def fetch_tle(identifier: str) -> dict:
    """
    Universal TLE fetcher. Detects whether identifier is a NORAD ID or name.

    Parameters
    ----------
    identifier : str  —  NORAD catalog number (numeric) or satellite name

    Returns
    -------
    dict: {name, line1, line2, norad_id, fetched_at, from_cache}
    """
    identifier = str(identifier).strip()
    if identifier.isdigit():
        return fetch_tle_by_norad(int(identifier))
    else:
        return fetch_tle_by_name(identifier)


def clear_cache(norad_id: int = None):
    """
    Clear cached TLE data.
    If norad_id is provided, clears only that entry; otherwise clears all.
    """
    if norad_id:
        path = _cache_path(f"norad_{norad_id}")
        if os.path.exists(path):
            os.remove(path)
    else:
        _ensure_cache_dir()
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(CACHE_DIR, f))
                except OSError:
                    pass
