import paho.mqtt.client as mqtt
import numpy as np
import joblib
import json
import collections
import asyncio
import websockets
import threading
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MQTT_HOST  = "7b89c60e7acf482d887aee8ae4be2830.s1.eu.hivemq.cloud"
MQTT_PORT  = 8883
MQTT_USER  = "kamikaze"
MQTT_PASS  = "Rudra@123"
MQTT_TOPIC = "running/imu"

WINDOW_SIZE   = 20    # rows per classification window
WS_PORT       = 8765  # browser dashboard connects here

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model        = joblib.load("zone_classifier.pkl")
FEATURE_COLS = joblib.load("feature_cols.pkl")
print("Model loaded — 98.7% accuracy")
print(f"Features: {FEATURE_COLS}\n")

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
buffer         = collections.deque(maxlen=WINDOW_SIZE)
latest_result  = {
    "zone":       "waiting",
    "confidence": 0.0,
    "probabilities": {},
    "pace":       "—",
    "suggestion": "—",
    "history":    []
}
target_zone    = "aerobic"
zone_history   = []   # list of last 20 predictions
ws_clients     = set()

ZONE_PACE = {
    "zone2":   "7:00–9:00 min/km",
    "aerobic": "5:30–7:00 min/km",
    "tempo":   "4:30–5:30 min/km",
    "intense": "3:45–4:30 min/km",
}
ZONE_ORDER = ["zone2", "aerobic", "tempo", "intense"]

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# matches MATLAB Phase 1 exactly:
# mean, std, rms per axis + sma_acc + sma_gyro
# ─────────────────────────────────────────────
def extract_features(rows):
    axes = ["aX","aY","aZ","gX","gY","gZ"]
    data = {ax: np.array([r[ax] for r in rows]) for ax in axes}

    feats = []
    for ax in axes:
        feats.append(float(np.mean(data[ax])))
    for ax in axes:
        feats.append(float(np.std(data[ax])))
    for ax in axes:
        feats.append(float(np.sqrt(np.mean(data[ax]**2))))

    sma_acc  = float(np.mean(
        np.abs(data["aX"]) + np.abs(data["aY"]) + np.abs(data["aZ"])))
    sma_gyro = float(np.mean(
        np.abs(data["gX"]) + np.abs(data["gY"]) + np.abs(data["gZ"])))
    feats.append(sma_acc)
    feats.append(sma_gyro)
    return feats

# ─────────────────────────────────────────────
# PACE SUGGESTION
# ─────────────────────────────────────────────
def get_suggestion(current, target):
    ci = ZONE_ORDER.index(current) if current in ZONE_ORDER else -1
    ti = ZONE_ORDER.index(target)  if target  in ZONE_ORDER else -1
    if ci == -1:
        return "Collecting data..."
    if ci == ti:
        return f"Hold pace — you are in {target}"
    elif ci < ti:
        return f"Speed up to reach {target}"
    else:
        return f"Slow down to reach {target}"

# ─────────────────────────────────────────────
# CLASSIFY
# ─────────────────────────────────────────────
def classify_window():
    global latest_result, zone_history

    feats = extract_features(list(buffer))
    X     = np.array(feats).reshape(1, -1)

    zone       = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    confidence = float(max(proba)) * 100
    probs_dict = {cls: round(float(p)*100, 1)
                  for cls, p in zip(model.classes_, proba)}

    zone_history.append(zone)
    if len(zone_history) > 20:
        zone_history = zone_history[-20:]

    suggestion = get_suggestion(zone, target_zone)

    latest_result = {
        "zone":          zone,
        "confidence":    round(confidence, 1),
        "probabilities": probs_dict,
        "pace":          ZONE_PACE.get(zone, "—"),
        "suggestion":    suggestion,
        "history":       zone_history[-20:],
        "target":        target_zone,
        "buffer_fill":   len(buffer),
        "window_size":   WINDOW_SIZE,
        "timestamp":     int(time.time() * 1000)
    }

    print(f"\nZONE: {zone.upper():10}  conf: {confidence:.1f}%  "
          f"suggestion: {suggestion}")

# ─────────────────────────────────────────────
# MQTT CALLBACKS
# ─────────────────────────────────────────────
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("MQTT connected!")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to: {MQTT_TOPIC}\n")
    else:
        print(f"MQTT failed: {reason_code}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        buffer.append(data)
        print(f"  buf [{len(buffer):02d}/{WINDOW_SIZE}] "
              f"aZ={data.get('aZ',0):.2f}", end='\r')

        if len(buffer) == WINDOW_SIZE:
            classify_window()
            # slide — remove oldest half
            for _ in range(WINDOW_SIZE // 2):
                buffer.popleft()

    except Exception as e:
        print(f"Message error: {e}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"\nMQTT disconnected: {reason_code}")

# ─────────────────────────────────────────────
# WEBSOCKET SERVER  (browser connects here)
# ─────────────────────────────────────────────
async def ws_handler(websocket):
    ws_clients.add(websocket)
    print(f"Browser connected  (total: {len(ws_clients)})")
    try:
        async for raw in websocket:
            # browser can send target zone
            try:
                msg = json.loads(raw)
                global target_zone
                if "target" in msg:
                    target_zone = msg["target"]
                    print(f"Target zone set to: {target_zone}")
            except:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"Browser disconnected (total: {len(ws_clients)})")

async def broadcast_loop():
    """Push latest result to all browsers every second."""
    global ws_clients  # Add this line
    while True:
        if ws_clients:
            payload = json.dumps(latest_result)
            dead = set()
            for ws in ws_clients.copy():
                try:
                    await ws.send(payload)
                except:
                    dead.add(ws)
            ws_clients -= dead
        await asyncio.sleep(1)

async def ws_main():
    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        print(f"WebSocket server started on ws://localhost:{WS_PORT}")
        await broadcast_loop()

def start_ws_server():
    asyncio.run(ws_main())

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # start WebSocket in background thread
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()
    time.sleep(1)

    # start MQTT
    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqttc.username_pw_set(MQTT_USER, MQTT_PASS)
    mqttc.tls_set()
    mqttc.on_connect    = on_connect
    mqttc.on_message    = on_message
    mqttc.on_disconnect = on_disconnect

    print("Connecting to MQTT broker...")
    mqttc.connect(MQTT_HOST, MQTT_PORT, keepalive=60)

    print("Open twin.html in your browser.\n")

    try:
        mqttc.loop_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        mqttc.disconnect()
