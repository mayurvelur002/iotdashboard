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
ILLUMINATION_THRESHOLD = 200
MAX_SCALE_LUX = 1000
CYAN = "#22d3ee"
PURPLE = "#c084fc"
GREEN = "#34d399"
RED = "#fb7185"

# -------------------- SESSION STATE ---------------------
defaults = {
    "history": [],
    "alert_log": [],
    "previous_alert": None,
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
    threshold_anomaly = illumination_value < ILLUMINATION_THRESHOLD
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


def create_gauge(value, title, threshold, accent):
    safe_value = max(0, min(float(value), MAX_SCALE_LUX))
    bar_color = GREEN if value >= threshold else RED
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_value,
            title={"text": title, "font": {"size": 15, "color": "#dcecff"}},
            number={"suffix": " lux", "font": {"size": 27, "color": "#f4faff"}},
            gauge={
                "axis": {"range": [0, MAX_SCALE_LUX], "tickcolor": "#8fa9c5"},
                "bar": {"color": bar_color, "thickness": 0.24},
                "bgcolor": "#10243a",
                "borderwidth": 1,
                "bordercolor": "rgba(34,211,238,.25)",
                "steps": [
                    {"range": [0, threshold], "color": "rgba(251,113,133,.18)"},
                    {"range": [threshold, MAX_SCALE_LUX], "color": "rgba(52,211,153,.12)"},
                ],
                "threshold": {
                    "line": {"color": accent, "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin={"t": 45, "b": 10, "l": 20, "r": 20},
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
      <div class="hero-sub">Real-Time Illumination Telemetry &amp; Threshold Supervision</div>
    </div>
    """,
    unsafe_allow_html=True,
)

thresh_1, thresh_2, thresh_3 = st.columns(3)
with thresh_1:
    st.metric("Minimum Safe Threshold", f"{ILLUMINATION_THRESHOLD} lux")
with thresh_2:
    st.metric("Warning Trigger Condition", f"< {ILLUMINATION_THRESHOLD} lux")
with thresh_3:
    st.metric("Full Scale Range", f"0 – {MAX_SCALE_LUX} lux")

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

is_normal = illumination_lux >= ILLUMINATION_THRESHOLD
low_light_alert = not is_normal

if low_light_alert != st.session_state.previous_alert:
    if low_light_alert:
        event_name = "EMERGENCY ALERT"
        severity = "CRITICAL"
        detection_method = "Threshold-Based Detection"
        details = f"Illumination ({illumination_lux} lux) below {ILLUMINATION_THRESHOLD} lux threshold"
    else:
        event_name = "SYSTEM RECOVERED"
        severity = "NORMAL"
        detection_method = "System Monitoring"
        details = f"Illumination ({illumination_lux} lux) at or above {ILLUMINATION_THRESHOLD} lux threshold"

    st.session_state.alert_log.append({
        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Event": event_name,
        "Severity": severity,
        "Detection Method": detection_method,
        "Illumination (lux)": illumination_lux,
        "Details": details,
    })
    st.session_state.alert_log = st.session_state.alert_log[-50:]
    st.session_state.previous_alert = low_light_alert

if not low_light_alert:
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
                f"Threshold detection: illumination is below {ILLUMINATION_THRESHOLD} lux."
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
        st.metric("Safety Threshold", f"{ILLUMINATION_THRESHOLD} lux")
        if is_normal:
            st.success("MONITOR STATUS: NORMAL")
        else:
            st.error("MONITOR STATUS: LOW ILLUMINATION")

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
                y=ILLUMINATION_THRESHOLD,
                line_dash="dash",
                line_color=RED,
                annotation_text=f"Threshold ({ILLUMINATION_THRESHOLD} lux)",
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
    st.subheader("Illumination Gauge")
    st.plotly_chart(
        create_gauge(
            illumination_lux,
            "ILLUMINATION LEVEL",
            ILLUMINATION_THRESHOLD,
            CYAN,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# -------------------- EMERGENCY PANEL --------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">04 / Emergency control</div>',
                unsafe_allow_html=True)
    st.subheader("Emergency Alert Control Center")
    if low_light_alert:
        st.error(
            f"🚨 CRITICAL WARNING: ILLUMINATION BELOW {ILLUMINATION_THRESHOLD} LUX"
        )
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
            st.info("Alarm muted. Visual emergency alerts remain active.")
            if st.button("🔊 Re-enable Alarm"):
                st.session_state.alarm_muted = False
                st.rerun()
    else:
        st.success("ALL SYSTEMS NORMAL — NO ACTIVE THRESHOLD ALERT")
        st.caption(
            f"Current illumination is at or above the {ILLUMINATION_THRESHOLD} lux threshold."
        )

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
