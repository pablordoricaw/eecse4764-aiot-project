# ==============================================================================
# MOCK DATABASE SERVER (Quiet Mode)
# Only prints BIG alerts when an error occurs.
# ==============================================================================
from flask import Flask, request, jsonify
from datetime import datetime, timezone
import random
import sys

app = Flask(__name__)

# CONFIGURATION
PORT = 8000
DEVICE_ID = "ventilator-01"

def get_utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def generate_random_log(req_device_id):
    # CHANGED: Lowered error chance to 5% so you have time to react
    is_error = random.random() < 0.05 
    
    timestamp = get_utc_now_iso()
    
    if is_error:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "ERROR",
            "event_type": "ERROR_EVENT",
            "error_code": "E-TEMP-HIGH", # Force the Overheat error for demo
            "message": "Critical fault: Temperature threshold exceeded."
        }
    else:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "INFO",
            "event_type": "MEASUREMENT",
            "error_code": None,
            "message": f"System nominal. Cycle: {random.randint(100,999)}"
        }

@app.route('/logs', methods=['GET'])
def get_logs():
    device_id = request.args.get('device_id', 'unknown')
    
    new_log = generate_random_log(device_id)
    
    response = {
        "logs": [new_log],
        "next_since": new_log['timestamp']
    }

    # === VISUAL IMPROVEMENT ===
    if new_log['level'] == 'ERROR':
        # Print a massive banner when an error happens
        print("\n" + "!"*60)
        print(f" >>> SENDING ERROR: {new_log['error_code']}")
        print(f" >>> SHINE FLASHLIGHT NOW!")
        print("!"*60 + "\n")
    else:
        # Just print a dot for normal info to keep terminal clean
        print(".", end="", flush=True)

    return jsonify(response)

if __name__ == '__main__':
    # 0.0.0.0 allows external access
    print(f"Mock DB running on Port {PORT}...")
    print("Waiting for ESP32...")
    app.run(host='0.0.0.0', port=PORT)