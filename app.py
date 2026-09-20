"""
app.py
------
Main Streamlit application for the Satellite Position Prediction ML System.
All times are displayed and accepted in IST (India Standard Time, UTC+5:30).

Sections:
    1. Search Panel         — satellite lookup by name or NORAD ID
    2. Live Position        — metrics + world map + ground track
    3. ML Prediction Panel  — future prediction
    4. Model Accuracy       — metrics table + actual vs predicted charts
    5. Satellite Info       — TLE details, cache status
"""

import sys
import os

# Ensure the satellite_predictor directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

import math
import time
from datetime import datetime, timezone, timedelta, date

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from tle_fetcher import fetch_tle
from sgp4_propagator import compute_position, get_tle_epoch_datetime
from feature_engineering import TARGET_COLS
from ml_model import (
    train_models,
    predict_position,
    load_metrics,
    load_actuals,
    load_model_name,
    is_model_trained,
)
from utils import (
    combine_date_time_ist,
    format_datetime_ist,
    format_datetime_utc,
    lat_str,
    lon_str,
    visibility_emoji,
    format_speed,
    format_period,
    split_ground_track_at_antimeridian,
    tle_epoch_age_hours,
    tle_age_warning,
    safe_round,
    r2_color,
    now_utc,
    now_ist,
    IST,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🛰️ Satellite Position Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://celestrak.org",
        "Report a Bug": None,
        "About": "## Satellite Position Predictor\nML-powered orbital prediction system.",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1829 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a2744 0%, #0f2027 50%, #1a2744 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(99,179,237,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed 0%, #90cdf4 50%, #63b3ed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #90cdf4;
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 300;
        letter-spacing: 0.05em;
    }

    /* Section cards */
    .section-card {
        background: rgba(26, 39, 68, 0.7);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #63b3ed;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric override */
    [data-testid="metric-container"] {
        background: rgba(15, 32, 55, 0.8);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 10px;
        padding: 16px !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(99,179,237,0.5);
    }
    [data-testid="stMetricValue"] {
        color: #90cdf4 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1829 0%, #0a1120 100%);
        border-right: 1px solid rgba(99,179,237,0.1);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #63b3ed;
    }

    /* Sidebar quick-search buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #1a4e8a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(49,130,206,0.4);
    }

    /* Code block */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0,0,0,0.3);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85em;
        color: #90cdf4;
    }

    /* Divider */
    hr {
        border-color: rgba(99,179,237,0.15);
        margin: 20px 0;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(15, 32, 55, 0.8) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: rgba(15, 32, 55, 0.8) !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        color: #e2e8f0 !important;
    }

    /* Tab style */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 32, 55, 0.5);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #718096;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(49,130,206,0.3) !important;
        color: #90cdf4 !important;
    }

    /* Status badges */
    .badge-visible {
        background: rgba(0,255,136,0.15);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-below {
        background: rgba(255,68,68,0.15);
        color: #ff6666;
        border: 1px solid rgba(255,68,68,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialization ──────────────────────────────────────────────

def _init_state():
    defaults = {
        "tle_data": None,
        "position_result": None,
        "search_history": [],
        "training_result": None,
        "prediction_result": None,
        "last_searched": None,
        "sat_id_value": "",       # controls text input value
        "auto_compute": False,    # flag to trigger auto-compute after sidebar click
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0;">
            <div style="font-size:3rem;">🛰️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#63b3ed; letter-spacing:0.05em;">
                SATELLITE PREDICTOR
            </div>
            <div style="font-size:0.75rem; color:#718096; margin-top:4px;">
                ML-Powered Orbital Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📡 Quick Search")
    st.caption("Click any satellite to load it instantly:")

    quick_sats = {
        "🌍 ISS (ZARYA)": "25544",
        "🌌 Hubble Space Telescope": "20580",
        "🔭 Chandra X-Ray Observatory": "25867",
        "🛸 Tiangong Space Station": "48274",
        "🌎 Terra (NASA EOS)": "25994",
        "🛰️ NOAA-19": "33591",
    }

    for label, norad in quick_sats.items():
        if st.button(label, key=f"quick_{norad}", use_container_width=True):
            # Store the NORAD ID and set auto_compute flag
            st.session_state["sat_id_value"] = norad
            st.session_state["auto_compute"] = True
            st.rerun()

    st.divider()

    # Search history
    if st.session_state["search_history"]:
        st.markdown("### 🕐 Recent Searches")
        for idx, hist_item in enumerate(st.session_state["search_history"][-5:][::-1]):
            st.caption(f"• {hist_item}")

    st.divider()
    st.markdown("### 🔗 Resources")
    st.markdown(
        """
        - [CelesTrak](https://celestrak.org)
        - [Space-Track.org](https://www.space-track.org)
        - [Heavens-Above](https://www.heavens-above.com)
        - [NASA Satellite Tracker](https://spotthestation.nasa.gov)
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 📚 Satellite Catalog Reference")
    st.caption("Copy any NORAD number and paste it in the search box above.")

    with st.expander("🏠 Space Stations"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `25544` | ISS (ZARYA) |
| `48274` | Tiangong Space Station |
""")

    with st.expander("🔭 Space Telescopes"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `20580` | Hubble Space Telescope |
| `25867` | Chandra X-Ray Observatory |
| `25989` | XMM-Newton (ESA) |
""")

    with st.expander("🌦️ Weather Satellites"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `33591` | NOAA-19 |
| `43013` | NOAA-20 |
| `54224` | GOES-18 |
| `38771` | Metop-B |
| `43689` | Metop-C |
""")

    with st.expander("🌍 Earth Observation"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `25994` | Terra (NASA EOS) |
| `27424` | Aqua (NASA) |
| `37849` | Suomi NPP |
| `40697` | Sentinel-2A (ESA) |
| `42063` | Sentinel-2B (ESA) |
| `39084` | Landsat-8 |
| `49260` | Landsat-9 |
""")

    with st.expander("🇮🇳 ISRO Satellites"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `37387` | RESOURCESAT-2 |
| `29710` | CARTOSAT-2 |
| `35931` | OCEANSAT-2 |
| `44233` | RISAT-2B |
| `39499` | INSAT-3D |
""")

    with st.expander("📍 GPS Satellites"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `26360` | GPS BIIR-2 |
| `29601` | GPS BIIR-11 |
| `43873` | GPS BIIIA-1 |
""")

    with st.expander("🔬 Science Missions"):
        st.markdown("""
| NORAD | Satellite |
|-------|-----------|
| `28376` | Aura (NASA) |
| `29107` | CloudSat |
| `43613` | ICESat-2 |
| `29108` | CALIPSO |
| `39159` | PROBA-V (ESA) |
| `39227` | KOMPSAT-3 |
""")


    st.divider()
    st.markdown(
        "<div style='color:#4a5568; font-size:0.7rem; text-align:center;'>"
        f"🇮🇳 IST: {now_ist().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )


# ── Hero Header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">🛰️ Satellite Position Predictor</div>
        <div class="hero-subtitle">
            Real-time orbital propagation · SGP4 · Skyfield · ML Position Forecasting · 🇮🇳 All times in IST
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SEARCH PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🔍 Section 1 — Satellite Search</div>', unsafe_allow_html=True)

# IST current time for defaults
_now_ist = now_ist()

with st.container():
    col_id, col_date, col_time, col_btn = st.columns([3, 2, 2, 1.5])

    with col_id:
        satellite_id = st.text_input(
            "🛰️ Satellite Name or NORAD ID",
            value=st.session_state["sat_id_value"],
            placeholder="e.g. ISS (ZARYA) or 25544",
            help="Enter the satellite name exactly as listed on CelesTrak, or its NORAD catalog number.",
            key="sat_id_input",
        )
        # Keep sat_id_value in sync with what user types
        if satellite_id != st.session_state["sat_id_value"]:
            st.session_state["sat_id_value"] = satellite_id

    with col_date:
        pos_date = st.date_input(
            "📅 Date (IST)",
            value=_now_ist.date(),
            key="pos_date",
        )

    with col_time:
        pos_time = st.time_input(
            "⏰ Time (IST)",
            value=_now_ist.time().replace(second=0, microsecond=0),
            key="pos_time",
            step=60,
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        compute_btn = st.button("🚀 Compute Position", key="compute_btn", use_container_width=True)


# ── Helper: fetch + compute ───────────────────────────────────────────────────

def _do_fetch_and_compute(sat_id: str, ist_date, ist_time):
    """Fetch TLE and compute position. target_dt is converted from IST to UTC internally."""
    target_dt_utc = combine_date_time_ist(ist_date, ist_time)

    with st.spinner(f"🔄 Fetching TLE data for **{sat_id}**..."):
        try:
            tle_data = fetch_tle(sat_id.strip())
        except ValueError as e:
            st.error(f"❌ Could not find satellite: {e}", icon="🚫")
            return
        except Exception as e:
            st.error(f"❌ Unexpected fetch error: {e}", icon="🔥")
            return

    st.session_state["tle_data"] = tle_data

    with st.spinner(f"⚙️ Computing position for **{tle_data['name']}**..."):
        try:
            position = compute_position(tle_data, target_dt_utc)
            st.session_state["position_result"] = position
            st.session_state["prediction_result"] = None  # clear old prediction
        except RuntimeError as e:
            st.error(f"❌ Computation error: {e}", icon="⚠️")
            return
        except Exception as e:
            st.error(f"❌ Unexpected compute error: {e}", icon="🔥")
            return

    # Update search history
    hist_entry = f"{tle_data['name']} (NORAD {tle_data.get('norad_id', '?')})"
    history = st.session_state["search_history"]
    if hist_entry not in history:
        history.append(hist_entry)
    st.session_state["search_history"] = history[-10:]
    st.session_state["last_searched"] = target_dt_utc

    ist_str = format_datetime_ist(target_dt_utc)
    st.success(
        f"✅ Position computed for **{tle_data['name']}** at {ist_str}",
        icon="✅",
    )


# ── Trigger: manual button OR auto (sidebar quick-search) ────────────────────

trigger_compute = compute_btn or st.session_state.get("auto_compute", False)

if trigger_compute:
    st.session_state["auto_compute"] = False   # reset flag
    _sat = st.session_state["sat_id_value"].strip()
    if _sat:
        _do_fetch_and_compute(_sat, pos_date, pos_time)
    else:
        st.warning("⚠️ Please enter a satellite name or NORAD ID before computing.", icon="⚠️")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE POSITION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-title">📡 Section 2 — Live Position Results</div>', unsafe_allow_html=True)

if st.session_state["position_result"] is None:
    st.info(
        "ℹ️ No position computed yet. Enter a satellite name or NORAD ID above and click **Compute Position**.",
        icon="ℹ️",
    )
else:
    pos = st.session_state["position_result"]
    tle = st.session_state["tle_data"]

    # Show which time was computed (in IST)
    if st.session_state["last_searched"]:
        ist_computed = format_datetime_ist(st.session_state["last_searched"])
        st.caption(f"📅 Computed for: **{ist_computed}**")

    # Row 1 — Primary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🌍 Latitude", lat_str(pos["latitude"]))
    with m2:
        st.metric("🌐 Longitude", lon_str(pos["longitude"]))
    with m3:
        st.metric("📏 Altitude", f"{pos['altitude_km']:,.1f} km")
    with m4:
        st.metric("⚡ Speed", f"{pos['speed_km_s']:.3f} km/s")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — Secondary metrics
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("🧭 Azimuth", f"{pos['azimuth_deg']:.1f}°")
    with m6:
        elev_delta = "+Above" if pos["elevation_deg"] > 0 else "-Below"
        st.metric("📐 Elevation", f"{pos['elevation_deg']:.1f}°", delta=elev_delta)
    with m7:
        st.metric("🔄 Orbital Period", format_period(pos["orbital_period_min"]))
    with m8:
        vis = pos["visibility"]
        vis_icon = "🟢" if "Visible" in vis else "🔴"
        st.metric("👁️ Visibility", f"{vis_icon} {vis}")

    # ── World Map ─────────────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Ground Track & Current Position")

    fig = go.Figure()

    # Ground track segments (split at anti-meridian)
    gt = pos.get("ground_track", {})
    gt_lats = gt.get("lats", [])
    gt_lons = gt.get("lons", [])

    if gt_lats and gt_lons:
        segments = split_ground_track_at_antimeridian(gt_lats, gt_lons)
        for seg_idx, (seg_lats, seg_lons) in enumerate(segments):
            fig.add_trace(go.Scattergeo(
                lat=seg_lats,
                lon=seg_lons,
                mode="lines",
                line=dict(color="rgba(99,179,237,0.5)", width=1.5),
                name="Orbit Path" if seg_idx == 0 else "",
                showlegend=(seg_idx == 0),
                hoverinfo="skip",
            ))

    # Current position marker
    fig.add_trace(go.Scattergeo(
        lat=[pos["latitude"]],
        lon=[pos["longitude"]],
        mode="markers+text",
        marker=dict(
            size=16,
            color="#00ff88",
            symbol="circle",
            line=dict(color="white", width=2),
        ),
        text=[tle.get("name", "SAT")],
        textposition="top center",
        textfont=dict(color="white", size=11),
        name="Current Position",
        hovertemplate=(
            f"<b>{tle.get('name', 'SAT')}</b><br>"
            f"Lat: {pos['latitude']:.4f}°<br>"
            f"Lon: {pos['longitude']:.4f}°<br>"
            f"Alt: {pos['altitude_km']:.1f} km<br>"
            f"Speed: {pos['speed_km_s']:.3f} km/s"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgba(20,40,70,0.9)",
            showocean=True,
            oceancolor="rgba(10,20,40,0.9)",
            showlakes=False,
            showcountries=True,
            countrycolor="rgba(99,179,237,0.2)",
            showcoastlines=True,
            coastlinecolor="rgba(99,179,237,0.3)",
            showframe=False,
            bgcolor="rgba(5,10,25,0.9)",
            projection_type="natural earth",
        ),
        paper_bgcolor="rgba(5,10,25,0.9)",
        plot_bgcolor="rgba(5,10,25,0.9)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        legend=dict(
            bgcolor="rgba(10,20,40,0.8)",
            bordercolor="rgba(99,179,237,0.3)",
            borderwidth=1,
            font=dict(color="#90cdf4"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ML PREDICTION PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-title">🤖 Section 3 — ML Position Prediction</div>', unsafe_allow_html=True)

if st.session_state["tle_data"] is None:
    st.info("ℹ️ Search and compute a satellite position first (Section 1) before running ML predictions.")
else:
    tle = st.session_state["tle_data"]

    col_train, col_pred = st.columns([1, 2])

    with col_train:
        st.markdown("**Step 1: Train the ML Model**")
        st.caption(
            "Trains on 7 days of SGP4-propagated position data for this satellite. "
            "Takes ~10-30 seconds depending on your hardware."
        )

        train_hours = st.slider(
            "Training window (hours)", min_value=24, max_value=336,
            value=168, step=24, key="train_hours",
            help="More hours = more training data = better model, but slower."
        )

        train_step = st.select_slider(
            "Sampling interval (minutes)", options=[2, 5, 10, 15, 30],
            value=5, key="train_step",
        )

        if st.button("🎯 Train Model", key="train_btn", use_container_width=True):
            with st.spinner("🔄 Generating training data and fitting models..."):
                try:
                    result = train_models(
                        tle_data=tle,
                        hours=train_hours,
                        step_minutes=train_step,
                    )
                    st.session_state["training_result"] = result
                    st.success(
                        f"✅ Model trained! {result['total_samples']:,} samples, "
                        f"best overall: **{result['overall_best']}**",
                        icon="✅",
                    )
                except ValueError as e:
                    st.error(f"Training failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected training error: {e}")

        if is_model_trained():
            st.caption(f"📦 Current model: **{load_model_name()}**")

    with col_pred:
        st.markdown("**Step 2: Predict Future Position**")
        st.caption("🇮🇳 Enter prediction date & time in IST")

        _tom_ist = (now_ist() + timedelta(days=1))
        col_pd, col_pt = st.columns(2)
        with col_pd:
            pred_date = st.date_input(
                "📅 Prediction Date (IST)",
                value=_tom_ist.date(),
                key="pred_date",
            )
        with col_pt:
            pred_time = st.time_input(
                "⏰ Prediction Time (IST)",
                value=now_ist().time().replace(second=0, microsecond=0),
                key="pred_time",
                step=60,
            )

        if st.button("🔮 Predict Position", key="predict_btn", use_container_width=True):
            if not is_model_trained():
                st.warning("⚠️ Train the model first (Step 1).", icon="⚠️")
            else:
                target_dt_utc = combine_date_time_ist(pred_date, pred_time)
                with st.spinner("🔄 Running ML prediction..."):
                    try:
                        pred = predict_position(tle, target_dt_utc)
                        st.session_state["prediction_result"] = {
                            **pred, "target_dt": target_dt_utc
                        }
                        st.success(
                            f"✅ Prediction complete for {format_datetime_ist(target_dt_utc)}",
                            icon="✅",
                        )
                    except RuntimeError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")

    # Display prediction results
    if st.session_state["prediction_result"]:
        pred = st.session_state["prediction_result"]
        st.markdown("---")
        target_ist = format_datetime_ist(pred.get("target_dt"))
        st.markdown(f"**🔮 Predicted Position at {target_ist}**")

        pm1, pm2, pm3, pm4 = st.columns(4)
        with pm1:
            st.metric("🌍 Predicted Latitude", lat_str(pred["latitude"]))
        with pm2:
            st.metric("🌐 Predicted Longitude", lon_str(pred["longitude"]))
        with pm3:
            st.metric("📏 Predicted Altitude", f"{pred['altitude_km']:,.1f} km")
        with pm4:
            st.metric("⚡ Predicted Speed", f"{pred['speed_km_s']:.3f} km/s")
        st.caption(f"🤖 Model used: **{pred.get('model_name', 'N/A')}**")

        # Show predicted position on map
        fig_pred = go.Figure()
        current_pos = st.session_state.get("position_result")
        if current_pos:
            gt = current_pos.get("ground_track", {})
            for seg_lats, seg_lons in split_ground_track_at_antimeridian(
                gt.get("lats", []), gt.get("lons", [])
            ):
                fig_pred.add_trace(go.Scattergeo(
                    lat=seg_lats, lon=seg_lons,
                    mode="lines",
                    line=dict(color="rgba(99,179,237,0.3)", width=1),
                    showlegend=False, hoverinfo="skip",
                ))
            fig_pred.add_trace(go.Scattergeo(
                lat=[current_pos["latitude"]], lon=[current_pos["longitude"]],
                mode="markers",
                marker=dict(size=12, color="#00ff88", symbol="circle",
                            line=dict(color="white", width=2)),
                name="Current Position",
                hovertemplate="<b>Current</b><br>Lat: %{lat:.4f}°<br>Lon: %{lon:.4f}°<extra></extra>",
            ))

        fig_pred.add_trace(go.Scattergeo(
            lat=[pred["latitude"]], lon=[pred["longitude"]],
            mode="markers+text",
            marker=dict(size=16, color="#f6ad55", symbol="star",
                        line=dict(color="white", width=2)),
            text=[f"Predicted\n{target_ist}"],
            textposition="top center",
            textfont=dict(color="white", size=10),
            name="Predicted Position",
            hovertemplate=(
                "<b>Predicted Position</b><br>"
                f"Lat: {pred['latitude']:.4f}°<br>"
                f"Lon: {pred['longitude']:.4f}°<br>"
                f"Alt: {pred['altitude_km']:.1f} km<br>"
                f"Speed: {pred['speed_km_s']:.3f} km/s"
                "<extra></extra>"
            ),
        ))

        fig_pred.update_layout(
            geo=dict(
                showland=True, landcolor="rgba(20,40,70,0.9)",
                showocean=True, oceancolor="rgba(10,20,40,0.9)",
                showcountries=True, countrycolor="rgba(99,179,237,0.2)",
                showcoastlines=True, coastlinecolor="rgba(99,179,237,0.3)",
                showframe=False, bgcolor="rgba(5,10,25,0.9)",
                projection_type="natural earth",
            ),
            paper_bgcolor="rgba(5,10,25,0.9)",
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
            legend=dict(bgcolor="rgba(10,20,40,0.8)", bordercolor="rgba(99,179,237,0.3)",
                        borderwidth=1, font=dict(color="#90cdf4")),
        )
        st.plotly_chart(fig_pred, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL ACCURACY PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-title">📊 Section 4 — Model Accuracy & Evaluation</div>', unsafe_allow_html=True)

metrics = load_metrics()
actuals = load_actuals()
model_name = load_model_name()

if metrics is None:
    st.info("ℹ️ No trained model found. Train a model in Section 3 to see accuracy metrics.")
else:
    tr = st.session_state.get("training_result") or {}

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("🤖 Best Model", model_name)
    with col_info2:
        total = tr.get("total_samples", "—")
        st.metric("📊 Training Samples", f"{total:,}" if isinstance(total, int) else total)
    with col_info3:
        test = tr.get("test_samples", "—")
        st.metric("🧪 Test Samples", f"{test:,}" if isinstance(test, int) else test)

    st.markdown("---")

    st.markdown("**📋 Evaluation Metrics per Target Variable**")

    TARGET_LABELS = {
        "latitude": "Latitude (°)",
        "longitude": "Longitude (°)",
        "altitude_km": "Altitude (km)",
        "speed_km_s": "Speed (km/s)",
    }

    table_rows = []
    for target in TARGET_COLS:
        for mname, mvals in metrics[target].items():
            table_rows.append({
                "Target": TARGET_LABELS.get(target, target),
                "Model": mname,
                "MAE": f"{mvals['mae']:.4f}",
                "RMSE": f"{mvals['rmse']:.4f}",
                "R²": f"{mvals['r2']:.4f}",
                "Quality": r2_color(mvals["r2"]),
            })

    df_metrics = pd.DataFrame(table_rows)
    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Target": st.column_config.TextColumn("Target Variable", width="medium"),
            "Model": st.column_config.TextColumn("Model", width="small"),
            "MAE": st.column_config.TextColumn("MAE", width="small"),
            "RMSE": st.column_config.TextColumn("RMSE", width="small"),
            "R²": st.column_config.TextColumn("R² Score", width="small"),
            "Quality": st.column_config.TextColumn("Quality", width="small"),
        },
    )

    # ── Model selection rationale ─────────────────────────────────────────────
    with st.expander("ℹ️ Model Selection Rationale"):
        st.markdown(
            f"""
            **Selected Model: {model_name}**

            The best model is selected per target variable based on the highest **R² score** 
            on the 20% hold-out test set. The model with the most target-wins is then used as 
            the overall predictor.

            | Model | Rationale |
            |-------|-----------|
            | **Random Forest** | Robust to outliers, handles non-linear relationships well, fast inference |
            | **XGBoost** | Gradient boosted trees, often achieves higher accuracy, especially on structured tabular data |

            The final model is a `MultiOutputRegressor` wrapper that trains one sub-estimator 
            per target variable (Latitude, Longitude, Altitude, Speed), allowing specialized 
            fitting for each output.

            **Feature Engineering**: The model uses 10 features derived from TLE orbital elements 
            (inclination, RAAN, eccentricity, BSTAR, mean motion, mean anomaly, argument of perigee, 
            epoch, revolution number) plus elapsed minutes from epoch as the primary time feature.
            """
        )

    # ── Actual vs Predicted charts ────────────────────────────────────────────
    if actuals is not None:
        st.markdown("**📈 Actual vs Predicted (Test Set)**")
        chart_cols = st.columns(2)
        target_items = list(actuals.items())

        for idx, (target, (y_test, y_pred)) in enumerate(target_items):
            col = chart_cols[idx % 2]
            with col:
                label = TARGET_LABELS.get(target, target)
                n = min(500, len(y_test))
                yt_sample = y_test[:n]
                yp_sample = y_pred[:n]

                fig_avp = go.Figure()
                fig_avp.add_trace(go.Scatter(
                    x=list(range(n)), y=yt_sample,
                    mode="lines",
                    name="Actual",
                    line=dict(color="#63b3ed", width=1.5),
                ))
                fig_avp.add_trace(go.Scatter(
                    x=list(range(n)), y=yp_sample,
                    mode="lines",
                    name="Predicted",
                    line=dict(color="#f6ad55", width=1.5, dash="dash"),
                ))
                fig_avp.update_layout(
                    title=dict(
                        text=f"<b>{label}</b>",
                        font=dict(color="#90cdf4", size=13),
                        x=0.01,
                    ),
                    paper_bgcolor="rgba(10,20,40,0.9)",
                    plot_bgcolor="rgba(5,10,25,0.9)",
                    margin=dict(l=40, r=10, t=40, b=30),
                    height=280,
                    xaxis=dict(
                        showgrid=True, gridcolor="rgba(99,179,237,0.1)",
                        color="#718096", title="Sample Index",
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="rgba(99,179,237,0.1)",
                        color="#718096", title=label,
                    ),
                    legend=dict(
                        bgcolor="rgba(10,20,40,0.8)", borderwidth=0,
                        font=dict(color="#90cdf4", size=11),
                        orientation="h", y=1.12,
                    ),
                )
                st.plotly_chart(fig_avp, use_container_width=True)

    # Retrain button
    st.markdown("---")
    if st.button("🔄 Retrain Model on Latest TLE Data", key="retrain_btn"):
        if st.session_state["tle_data"] is None:
            st.warning("⚠️ Fetch a satellite first (Section 1) before retraining.")
        else:
            with st.spinner("🔄 Retraining model..."):
                try:
                    result = train_models(
                        tle_data=st.session_state["tle_data"],
                        hours=168,
                        step_minutes=5,
                    )
                    st.session_state["training_result"] = result
                    st.success(
                        f"✅ Retrained! Best model: **{result['overall_best']}** "
                        f"on {result['total_samples']:,} samples.",
                        icon="✅",
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Retraining failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SATELLITE INFO PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-title">🗂️ Section 5 — Satellite Information</div>', unsafe_allow_html=True)

if st.session_state["tle_data"] is None:
    st.info("ℹ️ No satellite loaded yet. Use Section 1 to search for a satellite.")
else:
    tle = st.session_state["tle_data"]

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.metric("🛰️ Satellite Name", tle.get("name", "N/A"))
    with ic2:
        st.metric("🔢 NORAD ID", str(tle.get("norad_id", "N/A")))
    with ic3:
        epoch_dt = get_tle_epoch_datetime(tle)
        st.metric("📅 TLE Epoch (IST)", format_datetime_ist(epoch_dt) if epoch_dt else "N/A")

    # Cache status
    age_hours = tle_epoch_age_hours(tle)
    level, msg = tle_age_warning(age_hours)
    from_cache = tle.get("from_cache", False)

    col_cache, col_age = st.columns(2)
    with col_cache:
        if from_cache:
            st.success("📦 Data source: **Cache** (previously fetched)", icon="✅")
        else:
            st.info("🌐 Data source: **Fresh API call** (just fetched from CelesTrak)", icon="🌐")
    with col_age:
        if level == "success":
            st.success(msg, icon="✅")
        elif level == "warning":
            st.warning(msg, icon="⚠️")
        else:
            st.error(msg, icon="🔴")

    with st.expander("📋 Raw TLE Data", expanded=False):
        st.markdown(f"**Satellite:** `{tle.get('name', 'N/A')}`")
        st.markdown("**TLE Line 1:**")
        st.code(tle.get("line1", "N/A"), language="text")
        st.markdown("**TLE Line 2:**")
        st.code(tle.get("line2", "N/A"), language="text")
        fetched_at_ist = format_datetime_ist(
            datetime.fromisoformat(tle["fetched_at"]) if tle.get("fetched_at") else None
        )
        st.markdown(f"**Fetched at (IST):** {fetched_at_ist}")

    with st.expander("⚙️ Parsed Orbital Elements", expanded=False):
        try:
            from feature_engineering import parse_tle_elements
            elements = parse_tle_elements(tle)
            el_df = pd.DataFrame([{
                "Element": k,
                "Value": f"{v:.6f}" if isinstance(v, float) else str(v),
            } for k, v in elements.items() if k != "epoch_dt"])
            st.dataframe(el_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not parse orbital elements: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 16px 0;">
        🛰️ <strong style="color:#63b3ed;">Satellite Position Predictor</strong> &nbsp;|&nbsp;
        Powered by SGP4 · Skyfield · CelesTrak · Scikit-learn · XGBoost &nbsp;|&nbsp;
        🇮🇳 All times in <strong>IST (UTC+5:30)</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
