# ==============================================================================
# MOCK DATABASE SERVER (Quiet Mode)
# Only prints BIG alerts when an error occurs.
# Now supports both Temperature and Humidity data.
# ==============================================================================
from flask import Flask, request, jsonify
from datetime import datetime, timezone
import random
import sys

app = Flask(__name__)

# CONFIGURATION
PORT = 8000
DEVICE_ID = "ventilator-01"
ERROR_CHANCE = 0.15  # 15% chance of random error (increased from 5%)

# Simulated sensor state (for realistic trends)
current_temp = 25.0
current_humidity = 50.0
force_error = False  # Manual trigger flag

def get_utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def generate_random_log(req_device_id):
    global current_temp, current_humidity, force_error
    
    # Simulate gradual changes in temp and humidity
    current_temp += random.uniform(-0.5, 0.5)
    current_temp = max(15.0, min(45.0, current_temp))  # Keep between 15-45°C
    
    current_humidity += random.uniform(-1.0, 1.0)
    current_humidity = max(20.0, min(80.0, current_humidity))  # Keep between 20-80%
    
    # Check for forced error
    if force_error:
        force_error = False  # Reset flag
        is_error = True
        is_temp_error = True
        is_humidity_error = False
        is_random_error = False
    else:
        # Error conditions based on sensor values
        is_temp_error = current_temp > 40.0
        is_humidity_error = current_humidity > 75.0 or current_humidity < 25.0
        is_random_error = random.random() < ERROR_CHANCE  # Use configurable chance
        is_error = is_temp_error or is_humidity_error or is_random_error
    
    timestamp = get_utc_now_iso()
    
    if is_temp_error:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "ERROR",
            "event_type": "ERROR_EVENT",
            "error_code": "E-TEMP-HIGH",
            "message": f"Critical fault: Temperature threshold exceeded ({current_temp:.1f}°C)."
        }
    elif is_humidity_error:
        if current_humidity > 75.0:
            return {
                "timestamp": timestamp,
                "device_id": req_device_id,
                "level": "ERROR",
                "event_type": "ERROR_EVENT",
                "error_code": "E-HUMIDITY-HIGH",
                "message": f"Critical fault: Humidity too high ({current_humidity:.1f}%)."
            }
        else:
            return {
                "timestamp": timestamp,
                "device_id": req_device_id,
                "level": "ERROR",
                "event_type": "ERROR_EVENT",
                "error_code": "E-HUMIDITY-LOW",
                "message": f"Critical fault: Humidity too low ({current_humidity:.1f}%)."
            }
    elif is_random_error:
        error_types = [
            ("E-TEMP-HIGH", f"Critical fault: Temperature threshold exceeded ({current_temp:.1f}°C)."),
            ("E-HUMIDITY-HIGH", f"Critical fault: Humidity anomaly detected ({current_humidity:.1f}%)."),
            ("E-SENSOR-FAIL", "Critical fault: Sensor communication failure."),
        ]
        error_code, message = random.choice(error_types)
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "ERROR",
            "event_type": "ERROR_EVENT",
            "error_code": error_code,
            "message": message
        }
    else:
        return {
            "timestamp": timestamp,
            "device_id": req_device_id,
            "level": "INFO",
            "event_type": "MEASUREMENT",
            "error_code": None,
            "message": f"System nominal. Temp: {current_temp:.1f}°C, Humidity: {current_humidity:.1f}%. Cycle: {random.randint(100,999)}"
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
        print(f" >>> {new_log['message']}")
        print(f" >>> Current: Temp={current_temp:.1f}°C, Humidity={current_humidity:.1f}%")
        print("!"*60 + "\n")
    else:
        # Just print a dot for normal info to keep terminal clean
        print(".", end="", flush=True)

    return jsonify(response)

@app.route('/status', methods=['GET'])
def get_status():
    """Return current simulated sensor values"""
    return jsonify({
        "temperature": round(current_temp, 1),
        "humidity": round(current_humidity, 1),
        "timestamp": get_utc_now_iso()
    })

@app.route('/trigger-error', methods=['GET', 'POST'])
def trigger_error():
    """Manually trigger an error on next log request"""
    global force_error
    force_error = True
    print("\n" + "*"*60)
    print(" >>> ERROR TRIGGERED! Next log will be an ERROR.")
    print("*"*60 + "\n")
    return jsonify({"status": "Error will be sent on next log request"})

if __name__ == '__main__':
    # 0.0.0.0 allows external access
    print(f"Mock DB running on Port {PORT}...")
    print(f"Starting values: Temp={current_temp}°C, Humidity={current_humidity}%")
    print("Waiting for ESP32...")
    print("\nError triggers:")
    print(f"  - Random chance: {int(ERROR_CHANCE*100)}%")
    print("  - Temperature > 40°C")
    print("  - Humidity > 75% or < 25%")
    print(f"\nManual trigger: http://localhost:{PORT}/trigger-error")
    print("")
    app.run(host='0.0.0.0', port=PORT)