# ==============================================================================
# 1. Polls Mock Database for Logs (Port 8000)
# 2. Reads REAL Temperature from Si7021 Sensor
# 3. If Database sends ERROR -> Captures context -> Sends to LLM (Port 5001)
# ==============================================================================
from machine import Pin, I2C
import ssd1306
from si7021 import Si7021
import time
import network
import urequests
import ntptime

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
WIFI_SSID = "parzival"
WIFI_PASS = "82cmez3b3l18"

# SERVER CONFIGURATION (Using your Hotspot IP)
# Mock Database
DB_BASE_URL = "http://172.20.10.4:8000/logs"
DEVICE_ID = "ventilator-01"

# LLM Server
LLM_URL = "http://172.20.10.4:5001/api/ingest"

# RECORDING SETTINGS
PRE_ERROR_LOGS = 20
POST_ERROR_LOGS = 20
POLL_DELAY = 1.5  # Seconds between checks

# ------------------------------------------------------------------------------
# 2. HARDWARE SETUP (OLED + Si7021 Temperature Sensor)
# ------------------------------------------------------------------------------
# Enable power to STEMMA QT / I2C port (required for ESP32 Feather V2)
i2c_power = Pin(2, Pin.OUT)
i2c_power.value(1)
time.sleep(0.1)  # Give sensor time to power up

# I2C Bus - shared by OLED (0x3C) and Si7021 (0x40)
# Using pins: SDA=22, SCL=20 (same as your original code)
i2c = I2C(sda=Pin(22), scl=Pin(20))

# OLED Display
try:
    oled = ssd1306.SSD1306_I2C(128, 32, i2c)
    HAS_SCREEN = True
except:
    HAS_SCREEN = False
    print("[WARN] OLED not found")

# Si7021 Temperature & Humidity Sensor
try:
    temp_sensor = Si7021(i2c)
    HAS_TEMP_SENSOR = True
    print("[OK] Si7021 Temperature Sensor initialized")
except Exception as e:
    HAS_TEMP_SENSOR = False
    print(f"[WARN] Si7021 not found: {e}")


def read_temperature():
    """
    Reads REAL temperature from Si7021 sensor.
    Returns temperature in Celsius.
    Falls back to 20.0 if sensor not available.
    """
    if not HAS_TEMP_SENSOR:
        return 20.0
    try:
        temp_c = temp_sensor.read_temperature()
        return round(temp_c, 1)
    except Exception as e:
        print(f"[ERR] Temp read failed: {e}")
        return 20.0


def read_humidity():
    """
    Reads relative humidity from Si7021 sensor.
    Returns humidity in %.
    """
    if not HAS_TEMP_SENSOR:
        return 50.0
    try:
        rh = temp_sensor.read_humidity()
        return round(rh, 1)
    except Exception as e:
        print(f"[ERR] Humidity read failed: {e}")
        return 50.0


def update_display(line1, line2):
    if not HAS_SCREEN:
        return
    oled.fill(0)
    oled.text(line1, 0, 0, 1)
    oled.text(line2, 0, 12, 1)
    oled.show()


# ------------------------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------------------------
def sync_clock():
    print("[TIME] Syncing NTP...")
    try:
        ntptime.settime()
        print("[TIME] Synced!")
    except:
        print("[TIME] Failed (Will use relative time)")


def parse_iso_simple(iso_str):
    try:
        clean = iso_str.split(".")[0]
        parts = clean.replace("T", "-").replace(":", "-").split("-")
        t_tuple = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
            0,
            0,
        )
        return time.mktime(t_tuple)
    except:
        return 0


# ------------------------------------------------------------------------------
# 4. NETWORK LOGIC
# ------------------------------------------------------------------------------
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print(f"Connecting to {WIFI_SSID}...")
        wlan.connect(WIFI_SSID, WIFI_PASS)
        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
            print(".", end="")

    if wlan.isconnected():
        print(f"\n[WiFi] IP: {wlan.ifconfig()[0]}")
        return True
    return False


def fetch_logs(current_since):
    # Ask Mock DB for new logs
    url = f"{DB_BASE_URL}?device_id={DEVICE_ID}&limit=1&since={current_since}"
    try:
        res = urequests.get(url)
        if res.status_code == 200:
            data = res.json()
            res.close()
            return data.get("logs", []), data.get("next_since", current_since)
        res.close()
    except Exception as e:
        print(f"[GET] DB Error: {e}")
    return [], current_since


def send_package_to_llm(log_sequence, temp_history):
    print(f"\n[POST] Sending {len(log_sequence)} logs to LLM...")
    update_display("Packaging...", "Sending to LLM")

    # Filter temps to match log timeframe
    first_ts = parse_iso_simple(log_sequence[0].get("timestamp", ""))
    last_ts = parse_iso_simple(log_sequence[-1].get("timestamp", ""))

    relevant_temps = []
    for t in temp_history:
        # +/- 5 seconds buffer
        if t["ts"] >= (first_ts - 5) and t["ts"] <= (last_ts + 5):
            relevant_temps.append(t)

    payload = {
        "context": "Temperature and Humidity Event from Si7021 Sensor",
        "device_id": DEVICE_ID,
        "log_sequence": log_sequence,
        "temperature_data": relevant_temps,
    }

    headers = {"Content-Type": "application/json"}
    try:
        res = urequests.post(LLM_URL, json=payload, headers=headers)
        print(f"[POST] Status: {res.status_code}")
        res.close()
        return True
    except Exception as e:
        print(f"[POST] LLM Error: {e}")
        return False


# ------------------------------------------------------------------------------
# 5. MAIN LOOP
# ------------------------------------------------------------------------------
def main():
    if not connect_wifi():
        return
    sync_clock()

    # Scan I2C bus to verify devices
    print("[I2C] Scanning bus...")
    devices = i2c.scan()
    print(f"[I2C] Found devices: {[hex(d) for d in devices]}")
    # Expected: 0x3c (OLED) and 0x40 (Si7021)

    temp_history = []
    log_buffer = []

    state = "MONITORING"
    logs_needed_after_error = 0

    # Start looking for logs from roughly "now" (or a fixed past date to be safe)
    cursor_since = "2024-01-01T00:00:00.000Z"

    print("System Online. Reading REAL temperature from Si7021.")

    while True:
        # A. READ SENSOR (Real Temperature and Humidity from Si7021)
        current_temp = read_temperature()
        current_humidity = read_humidity()
        # Server expects: {"ts": float, "val": float, "humidity": float}
        temp_history.append(
            {"ts": time.time(), "val": current_temp, "humidity": current_humidity}
        )
        if len(temp_history) > 300:
            temp_history.pop(0)

        # B. FETCH LOGS (From Mock DB)
        new_logs, next_since = fetch_logs(cursor_since)
        if next_since:
            cursor_since = next_since

        # C. PROCESS LOGS
        for log in new_logs:
            log_buffer.append(log)
            print(f"[LOG] {log.get('message', '')}")

            if state == "MONITORING":
                # Keep buffer small
                if len(log_buffer) > PRE_ERROR_LOGS + 1:
                    log_buffer.pop(0)

                # TRIGGER: Check if Mock DB sent an error
                if log.get("error_code") is not None:
                    print("!!! ERROR DETECTED !!!")
                    state = "COLLECTING_POST_ERROR"
                    logs_needed_after_error = POST_ERROR_LOGS
                    update_display("ERROR!", "Capturing...")

            elif state == "COLLECTING_POST_ERROR":
                logs_needed_after_error -= 1
                update_display("Capturing...", f"Left: {logs_needed_after_error}")

                if logs_needed_after_error <= 0:
                    # DONE! Send to LLM
                    send_package_to_llm(log_buffer, temp_history)
                    log_buffer = []
                    state = "MONITORING"
                    update_display("Sent!", "Monitoring...")

        # Display Current Status
        if state == "MONITORING":
            update_display("Monitoring...", f"T:{current_temp}C H:{current_humidity}%")

        time.sleep(POLL_DELAY)


if __name__ == "__main__":
    main()

