import pandas as pd
import paho.mqtt.client as mqtt
import time
import json  # Added JSON library

# ─────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────
MQTT_HOST  = "7b89c60e7acf482d887aee8ae4be2830.s1.eu.hivemq.cloud"
MQTT_PORT  = 8883
MQTT_USER  = "kamikaze"
MQTT_PASS  = "Rudra@123"
MQTT_TOPIC = "running/imu"

# Set up MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.tls_set() 
client.connect(MQTT_HOST, MQTT_PORT)
client.loop_start()

print("Connected to HiveMQ. Starting Virtual Simulator...")

# Load your exact dataset
df = pd.read_csv("C:\\Users\\as020\\Downloads\\iot_aiml\\iot_aiml\\cleaned_dataset.csv")  # Update this path to your dataset

# ─────────────────────────────────────────────
# STREAM THE DATA (Now in JSON format)
# ─────────────────────────────────────────────
for index, row in df.iterrows():
    
    # Create a dictionary matching exactly what inference.py expects
    data_dict = {
        "aX": row['aX'],
        "aY": row['aY'],
        "aZ": row['aZ'],
        "gX": row['gX'],
        "gY": row['gY'],
        "gZ": row['gZ']
    }
    
    # Convert the dictionary into a JSON string
    payload = json.dumps(data_dict)
    
    # Publish to the broker
    client.publish(MQTT_TOPIC, payload)
    
    print(f"Streaming Row {index}: {payload}")
    
    # Pause for 0.1 seconds (10Hz)
    time.sleep(0.1)

print("Simulation complete.")
client.loop_stop()