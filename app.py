import json
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Aerospace Illumination Monitor",
    page_icon="💡",
    layout="wide",
)

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "aero-mission-x/dsu/illumination/lux"

MODES = [
    "Lab / test bench",
    "Cabin: boarding / taxi",
    "Cabin: night cruise",
]


class MQTTLuxListener:
    def __init__(self, broker: str, port: int, topic: str):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.lock = threading.Lock()
        self.latest_lux = None
        self.last_update = None
        self.connected = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self._thread = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        with self.lock:
            self.connected = (reason_code == 0)
        if reason_code == 0:
            client.subscribe(self.topic)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        with self.lock:
            self.connected = False

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if "lux" in payload:
                val = float(payload["lux"])
                with self.lock:
                    self.latest_lux = val
                    self.last_update = time.time()
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_forever()
        except Exception:
            with self.lock:
                self.connected = False

    def snapshot(self):
        with self.lock:
            return self.latest_lux, self.last_update, self.connected


@st.cache_resource
def get_listener() -> MQTTLuxListener:
    listener = MQTTLuxListener(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC)
    listener.start()
    return listener


def play_alert_sound(severity: str, pulse_id: int):
    """Plays a short Web Audio alert tone in the browser when triggered."""
    freq = 880.0 if severity == "EMERGENCY" else 587.33
    duration = 0.28 if severity == "EMERGENCY" else 0.18
    components.html(
        f"""
        <script>
        (function() {{
            const pulseId = {pulse_id};
            if (window.__lastAlertPulse === pulseId) return;
            window.__lastAlertPulse = pulseId;
            try {{
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                const ctx = new AudioCtx();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = "sine";
                osc.frequency.setValueAtTime({freq}, ctx.currentTime);
                gain.gain.setValueAtTime(0.18, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + {duration});
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + {duration});
            }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def classify_raw_state(lux: float, mode: str) -> tuple[str, str]:
    """Returns (STATE, reason) aligned with aerospace lighting threshold bands."""
    if mode == "Lab / test bench":
        if 500.0 <= lux <= 1000.0:
            return "NORMAL", "Within 500–1000 lux task band (EN 12464-1 / CIBSE SLL Guide 1)"
        if (300.0 <= lux < 500.0) or (1000.0 < lux <= 1500.0):
            detail = "Too dim (< 500 lux)" if lux < 500.0 else "High illuminance (> 1000–1500 lux)"
            return "WARNING", detail
        return "EMERGENCY", "Outside lab limits (< 300 lux or > 1500 lux)"

    if mode == "Cabin: boarding / taxi":
        if 300.0 <= lux <= 500.0:
            return "NORMAL", "Within 300–500 lux boarding/taxi design band"
        if (150.0 <= lux < 300.0) or (500.0 < lux <= 700.0):
            detail = "Too dim (< 300 lux)" if lux < 300.0 else "Too bright (> 500–700 lux)"
            return "WARNING", detail
        return "EMERGENCY", "Outside boarding/taxi limits (< 150 lux or > 700 lux)"

    # Mode 3: Cabin: night cruise
    if 0.0 <= lux <= 50.0:
        return "NORMAL", "Within 0–50 lux night-cruise comfort band"
    if 50.0 < lux <= 100.0:
        return "WARNING", "Above 50 lux (may disturb sleep physiology)"
    return "EMERGENCY", "Exceeds 100 lux night-cruise ceiling"


def apply_filtering(
    raw_state: str,
    debounce_s: float,
    persistence_s: float,
    hysteresis_pct: float,
) -> tuple[str, str]:
    now = time.time()
    ss = st.session_state

    if "committed_state" not in ss:
        ss.committed_state = raw_state
        ss.candidate_state = raw_state
        ss.candidate_since = now
        ss.warning_since = now if raw_state == "WARNING" else None

    hysteresis_note = f"Hysteresis ±{hysteresis_pct:.1f}%"
    if raw_state != ss.committed_state:
        if ss.candidate_state != raw_state:
            ss.candidate_state = raw_state
            ss.candidate_since = now

        elapsed_candidate = now - ss.candidate_since
        if elapsed_candidate >= debounce_s:
            if ss.committed_state == "WARNING" and raw_state == "EMERGENCY":
                if ss.warning_since is None:
                    ss.warning_since = now
                if (now - ss.warning_since) >= persistence_s:
                    ss.committed_state = "EMERGENCY"
                    ss.warning_since = None
                else:
                    rem = max(0.0, persistence_s - (now - ss.warning_since))
                    return ss.committed_state, f"WARNING escalating to EMERGENCY in {rem:.0f}s ({hysteresis_note})"
            else:
                ss.committed_state = raw_state
                ss.warning_since = now if raw_state == "WARNING" else None
    else:
        ss.candidate_state = raw_state
        ss.candidate_since = now
        if raw_state != "WARNING":
            ss.warning_since = None

    return ss.committed_state, f"Debounced ({debounce_s:.1f}s) • {hysteresis_note}"


def main():
    st.title("✈️ Aerospace Cabin & Lab Illumination Monitor")
    st.caption("Aligned with EN 12464-1, CIBSE SLL Guide 1, and cabin illumination bands")

    listener = get_listener()
    mqtt_lux, last_ts, mqtt_ok = listener.snapshot()

    with st.sidebar:
        st.header("Operating Mode & Telemetry")
        mode = st.selectbox("Operating Mode", MODES, index=0)
        live_mqtt = st.toggle("Receive live telemetry from Wokwi ESP32 via MQTT", value=True)
        sound_enabled = st.toggle("Enable Alert Sound", value=True)

        if live_mqtt:
            manual_lux = st.slider("Fallback / Test Lux (when no recent packet)", 0.0, 2000.0, 600.0, step=1.0)
        else:
            manual_lux = st.slider("Manual Lux Input", 0.0, 2000.0, 600.0, step=1.0)

        st.divider()
        st.subheader("Filtering & Timing (Design Values)")
        hysteresis_pct = st.slider("Boundary Hysteresis (%)", 0.0, 15.0, 5.0, step=0.5)
        debounce_s = st.slider("State Debounce (s)", 0.0, 10.0, 2.5, step=0.5)
        persistence_s = st.slider("WARNING → EMERGENCY Persistence (s)", 30, 180, 30, step=5)

    fresh_mqtt = live_mqtt and (mqtt_lux is not None) and (last_ts is not None) and ((time.time() - last_ts) <= 10.0)
    active_lux = float(mqtt_lux) if fresh_mqtt else float(manual_lux)

    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=120)
    st.session_state.history.append(active_lux)

    if "alert_pulse" not in st.session_state:
        st.session_state.alert_pulse = 0
    if "last_sound_ts" not in st.session_state:
        st.session_state.last_sound_ts = 0.0

    raw_state, reason = classify_raw_state(active_lux, mode)
    final_state, filter_info = apply_filtering(raw_state, debounce_s, persistence_s, hysteresis_pct)

    now = time.time()
    if sound_enabled and final_state in ("WARNING", "EMERGENCY") and (now - st.session_state.last_sound_ts >= 3.0):
        st.session_state.last_sound_ts = now
        st.session_state.alert_pulse += 1
        play_alert_sound(final_state, st.session_state.alert_pulse)

    if final_state == "NORMAL":
        st.success(f"🟢 **NORMAL** — {reason}  |  *{filter_info}*")
    elif final_state == "WARNING":
        st.warning(f"🟡 **WARNING** — {reason}  |  *{filter_info}*")
    else:
        st.error(f"🔴 **EMERGENCY** — {reason}  |  *{filter_info}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Illuminance", f"{active_lux:.2f} lux")
    c2.metric("Active State", final_state)
    c3.metric("Operating Mode", mode)
    c4.metric(
        "MQTT Broker Link",
        "Connected (Live)" if fresh_mqtt else ("Connected (Stale/Manual)" if mqtt_ok else "Manual / Offline"),
    )

    st.divider()

    st.subheader("Mode Threshold Reference")
    st.table(
        {
            "Mode": MODES,
            "🟢 NORMAL": [
                "500–1000 lux",
                "300–500 lux",
                "0–50 lux",
            ],
            "🟡 WARNING": [
                "300–<500 or >1000–1500 lux",
                "150–<300 or >500–700 lux",
                ">50–100 lux",
            ],
            "🔴 EMERGENCY": [
                "<300 or >1500 lux",
                "<150 or >700 lux",
                ">100 lux",
            ],
        }
    )

    st.subheader("Recent Illuminance Trend")
    st.line_chart(list(st.session_state.history))


if __name__ == "__main__":
    main()
