import paho.mqtt.client as mqtt
import json
import time

# MQTT Broker details
broker_address = "broker.emqx.io"  # or any other broker
port = 1883
topic = "urban_sounds/OE-007/data"

# Create data payload
payload = {
    "app_id": "urban_sounds_clap",
    "dev_id": "OE-007",
    "payload_fields": {
        "Talking": 0.500,
        "Silence": 0.2874208986759186,
        "Slamming door": 0.2655586302280426,
        "Noise": 0.0364396758377552,
        "Airco": 0.024280214682221413,
        "start_recording": int(time.time()),
        "RPI_temp": 54.9,
        "ptp": 0.19460979104042053
    },
    "time": time.time()
}

# Connect to the broker
client = mqtt.Client()
client.connect(broker_address, port)

# Publish the message
client.publish(topic, json.dumps(payload))
print(f"Message published to {topic}")

# Disconnect
client.disconnect()