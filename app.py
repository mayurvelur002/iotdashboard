import io
import math
import struct
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


# =========================================================
# AERO MISSION X | Software-only illumination simulation
# =========================================================

st.set_page_config(
    page_title="Aerospace Illumination Monitoring System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------- THEME -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #07111f; --panel: #0d1b2c; --cyan: #22d3ee;
        --purple: #c084fc; --muted: #9bb0c8;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
          radial-gradient(ellipse at 82% 0%, rgba(31,91,139,.20), transparent 34%),
          linear-gradient(135deg, #050d18 0%, #091626 55%, #07111f 100%);
        color: #edf6ff;
    }
    [data-testid="stHeader"], [data-testid="stSidebar"] { display: none; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }
    h1, h2, h3 { color: #f3f8ff !important; }
    p, label, .stCaption { color: #b9c9dc; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(16,36,58,.96), rgba(9,25,42,.96));
        border: 1px solid rgba(34,211,238,.32);
        border-radius: 13px; padding: 14px 16px;
        box-shadow: 0 0 18px rgba(34,211,238,.06);
    }
    div[data-testid="stMetric"] label { color: #a9bfd5 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f3f8ff !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(34,211,238,.27) !important;
        border-radius: 14px !important;
        background: linear-gradient(145deg, rgba(13,27,44,.92), rgba(8,20,34,.94));
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(34,211,238,.65); border-radius: 9px;
        background: linear-gradient(100deg, #0d3651, #142747);
        color: #ecfbff; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #22d3ee; color: white;
        box-shadow: 0 0 14px rgba(34,211,238,.25);
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background: #0b1c2e; border-color: rgba(34,211,238,.35);
    }
    hr { border-color: rgba(34,211,238,.22); }
    .small-label, .section-kicker {
        color: #22d3ee; font-size: .74rem; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .hero {
        padding: 18px 22px; border: 1px solid rgba(34,211,238,.35);
        border-radius: 16px;
        background: linear-gradient(105deg, rgba(12,34,55,.98), rgba(15,24,49,.94));
        box-shadow: 0 0 24px rgba(34,211,238,.07); margin-bottom: 14px;
    }
    .hero-title { font-size: 1.65rem; font-weight: 800; color: #f4faff; }
    .hero-sub { color: #a9bfd5; font-size: .9rem; margin-top: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- CONSTANTS -----------------------
CABIN_THRESHOLD = 100
LAB_THRESHOLD = 300
MAX_SCALE_LUX = 1000
CYAN = "#22d3ee"
PURPLE = "#c084fc"
GREEN = "#34d399"
RED = "#fb7185"
AMBER = "#fbbf24"

# -------------------- SESSION STATE ---------------------
defaults = {
    "history": [],
    "alert_log": [],
    "previous_alerts": [],
    "alarm_muted": False,
}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ------------------------- HELPERS ----------------------
def create_alarm_sound():
    sample_rate = 44100
    duration = 1.0
    frequency = 880
    amplitude = 0.25
    frames = []
    for index in range(int(sample_rate * duration)):
        time_value = index / sample_rate
        pulse = 1 if (time_value % 0.6) < 0.25 else 0
        sample = amplitude * math.sin(2 * math.pi * frequency * time_value) * pulse
        frames.append(struct.pack("<h", int(sample * 32767)))

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(b"".join(frames))
    return audio_buffer.getvalue()


def detect_anomaly(illumination_value):
    threshold_anomaly = (
        illumination_value < CABIN_THRESHOLD
        or illumination_value < LAB_THRESHOLD
    )
    ml_anomaly = False

    history = st.session_state.history
    if len(history) >= 10:
        training_df = pd.DataFrame(history)
        training_data = training_df[["Illumination"]].values
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        model.fit(training_data)
        ml_anomaly = model.predict([[illumination_value]])[0] == -1

    return {
        "threshold_anomaly": threshold_anomaly,
        "ml_anomaly": ml_anomaly,
        "combined_anomaly": threshold_anomaly or ml_anomaly,
    }


def style_plot(figure, height=320):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dcecff", "family": "Inter"},
        margin={"l": 18, "r": 18, "t": 55, "b": 18},
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        title_font={"size": 15, "color": "#edf6ff"},
    )
    figure.update_xaxes(gridcolor="rgba(148,170,195,.13)")
    figure.update_yaxes(gridcolor="rgba(148,170,195,.13)")
    return figure


def create_dual_threshold_gauge(value, cabin_thresh, lab_thresh):
    safe_value = max(0, min(float(value), MAX_SCALE_LUX))
    low_bound = min(cabin_thresh, lab_thresh)
    high_bound = max(cabin_thresh, lab_thresh)

    if value >= high_bound:
        bar_color = GREEN
    elif value >= low_bound:
        bar_color = AMBER
    else:
        bar_color = RED

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_value,
            title={
                "text": (
                    f"ILLUMINATION LEVEL "
                    f"(Cabin: {cabin_thresh} lx | Lab: {lab_thresh} lx)"
                ),
                "font": {"size": 14, "color": "#dcecff"},
            },
            number={"suffix": " lux", "font": {"size": 28, "color": "#f4faff"}},
            gauge={
                "axis": {"range": [0, MAX_SCALE_LUX], "tickcolor": "#8fa9c5"},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#10243a",
                "borderwidth": 1,
                "bordercolor": "rgba(34,211,238,.25)",
                "steps": [
                    {"range": [0, low_bound], "color": "rgba(251,113,133,.20)"},
                    {"range": [low_bound, high_bound], "color": "rgba(251,191,36,.15)"},
                    {"range": [high_bound, MAX_SCALE_LUX], "color": "rgba(52,211,153,.12)"},
                ],
                "threshold": {
                    "line": {"color": CYAN, "width": 3},
                    "thickness": 0.85,
                    "value": cabin_thresh,
                },
            },
        )
    )
    figure.add_annotation(
        x=0.5,
        y=0.06,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            f"<span style='color:{CYAN}'>● Cabin Threshold: {cabin_thresh} lux</span>  |  "
            f"<span style='color:{PURPLE}'>● Lab Threshold: {lab_thresh} lux</span>"
        ),
        font={"size": 12, "color": "#dcecff"},
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin={"t": 50, "b": 25, "l": 20, "r": 20},
        font={"color": "#dcecff"},
    )
    return figure


# --------------------- AUTO REFRESH ----------------------
st_autorefresh(interval=5000, limit=None, key="aero_refresh")

# ----------------------- HEADER & THRESHOLDS -----------------------
st.markdown(
    """
    <div class="hero">
      <div class="small-label">AEROSPACE ILLUMINATION MONITORING</div>
      <div class="hero-title">Aerospace Illumination Monitoring System</div>
      <div class="hero-sub">Real-Time Illumination Telemetry &amp; Dual-Threshold Supervision</div>
    </div>
    """,
    unsafe_allow_html=True,
)

thresh_1, thresh_2, thresh_3 = st.columns(3)
with thresh_1:
    st.metric("Aircraft Cabin Threshold", f"{CABIN_THRESHOLD} lux")
with thresh_2:
    st.metric("Aerospace Laboratory Threshold", f"{LAB_THRESHOLD} lux")
with thresh_3:
    st.metric("Gauge Scale Range", f"0 – {MAX_SCALE_LUX} lux")

# ----------------------- INPUT CONTROL -----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">01 / Input control</div>',
                unsafe_allow_html=True)
    st.subheader("Illumination Value Selector")
    illumination_lux = st.number_input(
        "Illumination level (lux)",
        min_value=0,
        max_value=2000,
        value=350,
        step=10,
        key="illumination_input",
    )

# Capture one sample each refresh interval
st.session_state.history.append({
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Illumination": illumination_lux,
})
st.session_state.history = st.session_state.history[-100:]

cabin_ok = illumination_lux >= CABIN_THRESHOLD
lab_ok = illumination_lux >= LAB_THRESHOLD

active_alerts = []
if not cabin_ok:
    active_alerts.append(
        f"Aircraft Cabin Alert: Illumination ({illumination_lux} lux) is below {CABIN_THRESHOLD} lux"
    )
if not lab_ok:
    active_alerts.append(
        f"Aerospace Laboratory Alert: Illumination ({illumination_lux} lux) is below {LAB_THRESHOLD} lux"
    )

if active_alerts != st.session_state.previous_alerts:
    if active_alerts:
        event_name = "EMERGENCY ALERT"
        severity = "CRITICAL"
        detection_method = "Dual-Threshold Detection"
        details = " | ".join(active_alerts)
    else:
        event_name = "SYSTEM RECOVERED"
        severity = "NORMAL"
        detection_method = "System Monitoring"
        details = (
            f"Illumination ({illumination_lux} lux) meets both "
            f"{CABIN_THRESHOLD} lux (Cabin) and {LAB_THRESHOLD} lux (Lab) thresholds"
        )

    st.session_state.alert_log.append({
        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Event": event_name,
        "Severity": severity,
        "Detection Method": detection_method,
        "Illumination (lux)": illumination_lux,
        "Details": details,
    })
    st.session_state.alert_log = st.session_state.alert_log[-50:]
    st.session_state.previous_alerts = active_alerts.copy()

if not active_alerts:
    st.session_state.alarm_muted = False

anomaly_result = detect_anomaly(illumination_lux)

# ---------------- REAL-TIME MONITOR + GRAPH ------------------
history_df = pd.DataFrame(st.session_state.history)

with st.container(border=True):
    st.markdown('<div class="section-kicker">02 / Real-time monitor</div>',
                unsafe_allow_html=True)
    st.subheader("Real-Time Monitor")

    if anomaly_result["combined_anomaly"]:
        st.error("🚨 ANOMALY DETECTED — REVIEW REQUIRED")
        if anomaly_result["threshold_anomaly"]:
            st.warning(
                f"Threshold detection: illumination ({illumination_lux} lux) is below "
                f"one or both thresholds (Cabin: {CABIN_THRESHOLD} lux, Lab: {LAB_THRESHOLD} lux)."
            )
        if anomaly_result["ml_anomaly"]:
            st.warning(
                "Isolation Forest: current reading differs from the learned historical pattern."
            )
    else:
        st.success("● AI MONITORING: NO ANOMALIES DETECTED")

    monitor_left, monitor_right = st.columns([1, 2])
    with monitor_left:
        st.metric("Current Illumination", f"{illumination_lux} lux")
        if cabin_ok:
            st.success(f"CABIN ({CABIN_THRESHOLD} lux): NORMAL")
        else:
            st.error(f"CABIN ({CABIN_THRESHOLD} lux): ALERT (< {CABIN_THRESHOLD} lux)")

        if lab_ok:
            st.success(f"LABORATORY ({LAB_THRESHOLD} lux): NORMAL")
        else:
            st.warning(f"LABORATORY ({LAB_THRESHOLD} lux): ALERT (< {LAB_THRESHOLD} lux)")

    with monitor_right:
        if not history_df.empty:
            live_figure = go.Figure()
            live_figure.add_trace(
                go.Scatter(
                    x=history_df["Time"],
                    y=history_df["Illumination"],
                    mode="lines+markers",
                    name="Illumination (lux)",
                    line={"color": CYAN, "width": 2.5},
                    marker={"size": 5},
                )
            )
            live_figure.add_hline(
                y=CABIN_THRESHOLD,
                line_dash="dash",
                line_color=CYAN,
                annotation_text=f"Cabin ({CABIN_THRESHOLD} lux)",
                annotation_position="bottom left",
            )
            live_figure.add_hline(
                y=LAB_THRESHOLD,
                line_dash="dot",
                line_color=PURPLE,
                annotation_text=f"Lab ({LAB_THRESHOLD} lux)",
                annotation_position="top left",
            )
            live_figure.update_layout(
                title="Real-Time Monitoring Graph",
                yaxis_title="Illumination (lux)",
            )
            st.plotly_chart(
                style_plot(live_figure, 320),
                use_container_width=True,
                config={"displayModeBar": False},
            )

# ------------------------- GAUGE -------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">03 / Instrument gauge</div>',
                unsafe_allow_html=True)
    st.subheader("Dual-Threshold Illumination Gauge")
    st.plotly_chart(
        create_dual_threshold_gauge(
            illumination_lux,
            CABIN_THRESHOLD,
            LAB_THRESHOLD,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        f"Cyan marker = Aircraft Cabin threshold ({CABIN_THRESHOLD} lux) · "
        f"Purple marker = Aerospace Laboratory threshold ({LAB_THRESHOLD} lux)"
    )

# -------------------- EMERGENCY PANEL --------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">04 / Emergency control</div>',
                unsafe_allow_html=True)
    st.subheader("Emergency Alert Control Center")

    alert_col_1, alert_col_2 = st.columns(2)
    with alert_col_1:
        st.markdown(f"**✈️ Aircraft Cabin Alert (< {CABIN_THRESHOLD} lux)**")
        if not cabin_ok:
            st.error(
                f"CABIN ALERT ACTIVE: {illumination_lux} lux is below the "
                f"{CABIN_THRESHOLD} lux cabin threshold."
            )
        else:
            st.success(f"CABIN CLEAR: {illumination_lux} lux ≥ {CABIN_THRESHOLD} lux")

    with alert_col_2:
        st.markdown(f"**⌁ Aerospace Laboratory Alert (< {LAB_THRESHOLD} lux)**")
        if not lab_ok:
            st.warning(
                f"LABORATORY ALERT ACTIVE: {illumination_lux} lux is below the "
                f"{LAB_THRESHOLD} lux laboratory threshold."
            )
        else:
            st.success(f"LABORATORY CLEAR: {illumination_lux} lux ≥ {LAB_THRESHOLD} lux")

    if active_alerts:
        if not st.session_state.alarm_muted:
            st.audio(
                create_alarm_sound(),
                format="audio/wav",
                autoplay=True,
                loop=True,
            )
            if st.button("🔇 Silence Alarm"):
                st.session_state.alarm_muted = True
                st.rerun()
        else:
            st.info("Alarm muted. Visual threshold alerts remain active.")
            if st.button("🔊 Re-enable Alarm"):
                st.session_state.alarm_muted = False
                st.rerun()
    else:
        st.success("ALL THRESHOLDS SATISFIED — NO ACTIVE ALERTS")

# ------------------------- FOOTER ------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#829bb5;font-size:.78rem'>"
    "AERO MISSION X · Aerospace Illumination Monitoring System<br>"
    "Software-based sensor simulation · Demonstration thresholds only · "
    "Not certified for aviation use"
    "</div>",
    unsafe_allow_html=True,
)
