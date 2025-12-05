from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import random
import time

app = Flask(__name__)

# CONFIGURATION
PORT = 8000
DEVICE_ID = "ventilator-01"

def get_utc_now_iso():
    """Returns current time in ISO-8601 format with milliseconds"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def generate_random_log(req_device_id):
    """Generates a single random log entry"""
    
    # 10% chance of an error, 90% chance of normal info
    is_error = random.random() < 0.1
    
    timestamp = get_utc_now_iso()
    
    if is_error:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "ERROR",
            "event_type": "ERROR_EVENT",
            "error_code": random.choice(["E-TEMP-HIGH", "E-PRESSURE-LOW", "E-SENSOR-FAIL"]),
            "message": "Critical fault detected in sensor array."
        }
    else:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "INFO",
            "event_type": "MEASUREMENT",
            "error_code": None, # Null for normal logs
            "message": f"Routine measurement check. Status: OK. Cycle: {random.randint(100,999)}"
        }

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Simulates: GET /logs?device_id=...&since=...&limit=...
    """
    # 1. Parse Query Parameters
    device_id = request.args.get('device_id', 'unknown')
    since = request.args.get('since', '')
    limit = int(request.args.get('limit', 10))

    print(f"\n[REQ] ESP32 asked for logs since: {since}")

    # 2. Logic to "fake" new data
    # In a real DB, we would query SQL. 
    # Here, we just generate a NEW log right now to feed the ESP32.
    
    # To prevent flooding, if the ESP32 asks too fast, we might return nothing sometimes.
    # But for testing, let's always give 1 new log.
    
    new_log = generate_random_log(device_id)
    
    # 3. Construct Response
    response = {
        "logs": [new_log],
        "next_since": new_log['timestamp'] # The pointer for the next request
    }

    # Print to console so you can see what's happening
    if new_log['level'] == 'ERROR':
        print(f"   >>> SENDING ERROR: {new_log['error_code']}")
    else:
        print(f"   >>> Sending Info: {new_log['message']}")

    return jsonify(response)

if __name__ == '__main__':
    # 0.0.0.0 allows external access (from ESP32)
    print(f"Mock Database running on port {PORT}...")
    print(f"Update your ESP32 'DB_BASE_URL' to: http://<YOUR_LAPTOP_IP>:{PORT}/logs")
    app.run(host='0.0.0.0', port=PORT)