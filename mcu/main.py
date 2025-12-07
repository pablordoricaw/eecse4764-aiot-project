"""
main.py

MicroPython client running on HUZZAH32 v2.

Responsibilities:
- Connect to Wi‑Fi using credentials from config.py.
- Periodically fetch log records from the logs server (/logs endpoint).
- Read real temperature and humidity from the Si7021 sensor over I2C.
- For error episodes, capture a window of pre/post logs, enrich each record
  with the closest-in-time sensor_temperature and sensor_humidity readings,
  and POST EACH ENRICHED LOG INDIVIDUALLY to the LLM server.
"""

import network
import ntptime
import time
import urequests
import ujson
import gc

from machine import Pin, I2C

import ssd1306
import config
from si7021 import Si7021

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
WIFI_SSID = config.WIFI_SSID
WIFI_PASS = config.WIFI_PASS

LOGS_SERVER_URL = config.LOGS_SERVER_URL
DEVICE_ID = config.DEVICE_ID

LLM_SERVER_URL = config.LLM_SERVER_URL

MAX_SENSOR_READINGS_BUFFER_SIZE = 300

LOGS_REQUEST_LIMIT = 20
PRE_ERROR_LOGS = 2
POST_ERROR_LOGS = 2

POLL_DELAY = 1.5  # Seconds between checks

# ------------------------------------------------------------------------------
# 2. HARDWARE SETUP (OLED + Si7021 Temperature Sensor)
# ------------------------------------------------------------------------------
MAX_DISPLAY_CHARS = 16  # 128 pixels / 8 px per char
LINE_HEIGHT = 8  # 8 px per text row

i2c_power = Pin(2, Pin.OUT)
i2c_power.value(1)
time.sleep(0.1)

i2c = I2C(sda=Pin(22), scl=Pin(20))

try:
    oled = ssd1306.SSD1306_I2C(128, 32, i2c)
    HAS_SCREEN = True
except:
    HAS_SCREEN = False
    print("[WARN] OLED not found")

try:
    temp_sensor = Si7021(i2c)
    HAS_TEMP_SENSOR = True
    print("[OK] Si7021 Temperature Sensor initialized")
except Exception as e:
    HAS_TEMP_SENSOR = False
    print(f"[WARN] Si7021 not found: {e}")


def read_temperature():
    if not HAS_TEMP_SENSOR:
        return 20.0
    try:
        temp_c = temp_sensor.read_temperature()
        return round(temp_c, 1)
    except Exception as e:
        print(f"[ERR] Temp read failed: {e}")
        return 20.0


def read_humidity():
    if not HAS_TEMP_SENSOR:
        return 50.0
    try:
        rh = temp_sensor.read_humidity()
        return round(rh, 1)
    except Exception as e:
        print(f"[ERR] Humidity read failed: {e}")
        return 50.0


def _format_line(text, align):
    if text is None:
        text = ""
    s = str(text)[:MAX_DISPLAY_CHARS]
    pad = MAX_DISPLAY_CHARS - len(s)

    if align == "right":
        return " " * pad + s
    elif align == "center":
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right
    else:
        return s + " " * pad


def write_lines_to_display(line1=None, line2=None, line3=None, line4=None):
    if not HAS_SCREEN:
        return

    oled.fill(0)
    lines = [line1, line2, line3, line4]
    for idx, spec in enumerate(lines):
        if spec is None:
            continue
        text, align = spec
        formatted = _format_line(text, align)
        y = idx * LINE_HEIGHT
        oled.text(formatted, 0, y, 1)
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
    url = f"{LOGS_SERVER_URL}?device_id={DEVICE_ID}&limit={LOGS_REQUEST_LIMIT}&since={current_since}"
    try:
        res = urequests.get(url, timeout=10)
        try:
            if res.status_code == 200:
                data = res.json()
                return data.get("logs", []), data.get("next_since", current_since)
            else:
                print(f"[GET] Logs Server HTTP {res.status_code}: {res.text}")
        finally:
            res.close()
    except Exception as e:
        print(f"[GET] Logs Server Error: {e}")
    return [], current_since


def find_closest_sensor_sample(log_timestamp_iso, sensor_readings_history):
    """
    Given a log timestamp (ISO string) and sensor_readings_history list of
    { "ts": epoch_seconds, "val": temp_c, "humidity": rh },
    return (sensor_temp, sensor_humidity) for the closest sample in time.
    """
    log_ts = parse_iso_simple(log_timestamp_iso)
    if log_ts == 0 or not sensor_readings_history:
        return 20.0, 50.0

    closest = sensor_readings_history[0]
    min_delta = abs(closest["ts"] - log_ts)

    for sample in sensor_readings_history[1:]:
        delta = abs(sample["ts"] - log_ts)
        if delta < min_delta:
            closest = sample
            min_delta = delta

    return closest["val"], closest["humidity"]


def send_single_log_to_llm(log_entry, index, total):
    """
    Send a single enriched log entry to the LLM server.
    """
    index += 1
    print(f"\n[POST] Sending log {index}/{total} to LLM server...")
    write_lines_to_display(
        ("Packaging...", "left"),
        (f"Log {index}/{total}", "left"),
        ("Sending to LLM", "left"),
    )

    payload = {
        "log_index": index,
        "log_count": total,
        "log": log_entry,
    }

    headers = {"Content-Type": "application/json"}

    try:
        gc.collect()
        body = ujson.dumps(payload)
        print(f"[POST] Payload length: {len(body)} bytes")
    except Exception as e:
        print(f"[POST] JSON encode error: {e}")
        return False

    try:
        res = urequests.post(LLM_SERVER_URL, data=body, headers=headers)
        print(f"[POST] Status: {res.status_code}")
        res.close()
        return True
    except Exception as e:
        print(f"[POST] LLM Server Error: {e}")
        return False


# ------------------------------------------------------------------------------
# 5. MAIN LOOP
# ------------------------------------------------------------------------------
def main():
    if not connect_wifi():
        return
    sync_clock()

    print("[I2C] Scanning bus...")
    devices = i2c.scan()
    print(f"[I2C] Found devices: {[hex(d) for d in devices]}")

    sensor_readings_history = []
    log_buffer = []

    state = "MONITORING"
    logs_needed_after_error = 0

    cursor_since = "2024-01-01T00:00:00.000Z"

    print("System Online. Reading REAL temperature from Si7021.")

    while True:
        # A. READ SENSOR
        current_temp = read_temperature()
        current_humidity = read_humidity()
        sensor_readings_history.append(
            {"ts": time.time(), "val": current_temp, "humidity": current_humidity}
        )
        if len(sensor_readings_history) > MAX_SENSOR_READINGS_BUFFER_SIZE:
            sensor_readings_history.pop(0)

        # B. FETCH LOGS
        new_logs, next_since = fetch_logs(cursor_since)
        if next_since:
            cursor_since = next_since

        # C. PROCESS LOGS
        for log in new_logs:
            log_buffer.append(log)
            ts = log.get("device_timestamp", "")
            msg = log.get("device_message", "")
            print(f"[LOG] {ts} - {msg}")

            if state == "MONITORING":
                if len(log_buffer) > PRE_ERROR_LOGS + 1:
                    log_buffer.pop(0)

                # Trigger on device-level error events
                if log.get("device_event_type") == "ERROR_EVENT":
                    print("!!! ERROR DETECTED !!!")
                    state = "COLLECTING_POST_ERROR"
                    logs_needed_after_error = POST_ERROR_LOGS
                    write_lines_to_display(("ERROR!", "left"), ("Capturing...", "left"))

            elif state == "COLLECTING_POST_ERROR":
                logs_needed_after_error -= 1
                write_lines_to_display(
                    ("Capturing...", "left"),
                    (f"Left: {logs_needed_after_error}", "left"),
                )

                if logs_needed_after_error <= 0:
                    # Enrich each log with closest sensor reading and send individually
                    total = len(log_buffer)
                    for idx, base_log in enumerate(log_buffer):
                        enriched = dict(base_log)
                        ts_iso = enriched.get("device_timestamp", "")
                        sensor_temp, sensor_humidity = find_closest_sensor_sample(
                            ts_iso, sensor_readings_history
                        )
                        enriched["sensor_temp_c"] = sensor_temp
                        enriched["sensor_humidity"] = sensor_humidity

                        send_single_log_to_llm(enriched, idx, total)

                    log_buffer = []
                    state = "MONITORING"
                    write_lines_to_display(("Sent!", "left"), ("Monitoring...", "left"))

        # Display Current Status
        if state == "MONITORING":
            write_lines_to_display(
                ("Monitoring...", "left"),
                (f"T:{current_temp}C H:{current_humidity}%", "right"),
            )

        time.sleep(POLL_DELAY)


if __name__ == "__main__":
    main()
