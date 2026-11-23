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


def classify_state(lux: float, mode: str) -> tuple[str, str]:
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


def main():
    st.title("✈️ Aerospace Cabin & Lab Illumination Monitor")
    st.caption("Aligned with EN 12464-1, CIBSE SLL Guide 1, and cabin illumination bands")

    listener = get_listener()
    mqtt_lux, last_ts, mqtt_ok = listener.snapshot()

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        mode = st.selectbox("Operating Mode", MODES, index=0)
    with ctrl_col2:
        live_mqtt = st.toggle("Live MQTT (Wokwi ESP32)", value=True)
    with ctrl_col3:
        sound_enabled = st.toggle("Enable Alert Sound", value=True)

    slider_label = (
        "Fallback / Test Lux (used when no recent MQTT packet arrives)"
        if live_mqtt
        else "Manual Lux Input"
    )
    manual_lux = st.slider(slider_label, min_value=0.0, max_value=2000.0, value=600.0, step=1.0)

    fresh_mqtt = (
        live_mqtt
        and (mqtt_lux is not None)
        and (last_ts is not None)
        and ((time.time() - last_ts) <= 10.0)
    )
    active_lux = float(mqtt_lux) if fresh_mqtt else float(manual_lux)

    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=120)
    if "last_recorded_ts" not in st.session_state:
        st.session_state.last_recorded_ts = None

    # Record history on new MQTT packets or when in manual mode
    if fresh_mqtt:
        if last_ts != st.session_state.last_recorded_ts:
            st.session_state.history.append(active_lux)
            st.session_state.last_recorded_ts = last_ts
    else:
        st.session_state.history.append(active_lux)

    if "alert_pulse" not in st.session_state:
        st.session_state.alert_pulse = 0
    if "last_sound_ts" not in st.session_state:
        st.session_state.last_sound_ts = 0.0

    state, reason = classify_state(active_lux, mode)

    now = time.time()
    if sound_enabled and state in ("WARNING", "EMERGENCY") and (now - st.session_state.last_sound_ts >= 3.0):
        st.session_state.last_sound_ts = now
        st.session_state.alert_pulse += 1
        play_alert_sound(state, st.session_state.alert_pulse)

    if state == "NORMAL":
        st.success(f"🟢 **NORMAL** — {reason}")
    elif state == "WARNING":
        st.warning(f"🟡 **WARNING** — {reason}")
    else:
        st.error(f"🔴 **EMERGENCY** — {reason}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Illuminance", f"{active_lux:.2f} lux")
    c2.metric("Active State", state)
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

    # Auto-refresh every 2 seconds when Live MQTT is active so new packets appear automatically
    if live_mqtt:
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
