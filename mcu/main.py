from machine import Pin, I2C, ADC
import ssd1306
import time
import network
import ujson
import urequests

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
WIFI_SSID = "Columbia University"
# WIFI_PASS = "" 

# DATABASE SETTINGS
DB_BASE_URL = "http://192.168.1.XXX:8000/logs"
DEVICE_ID   = "ventilator-01"
LOG_LIMIT   = 1  # Get 1 record at a time

# LLM SETTINGS
LLM_URL = "http://192.168.1.YYY:5000/api/ingest"

# Polling Interval (Seconds)
POLL_DELAY = 2

# ------------------------------------------------------------------------------
# 2. HARDWARE SETUP
# ------------------------------------------------------------------------------
i2c = I2C(sda=Pin(22), scl=Pin(20))
try:
    oled = ssd1306.SSD1306_I2C(128, 32, i2c)
    HAS_SCREEN = True
except:
    HAS_SCREEN = False

# Temperature Sensor (Simulated on Pin 34)
temp_adc = ADC(Pin(34))
temp_adc.atten(ADC.ATTN_11DB) 

def read_sensor_temperature():
    try:
        raw_val = temp_adc.read()
        # Simulated fluctuation around 73.0C
        return round(73.0 + ((raw_val % 20) / 10.0), 1)
    except:
        return 73.0

def update_display(status_line, temp_val):
    if not HAS_SCREEN: return
    oled.fill(0)
    oled.text("Stage 2: Bridge", 0, 0, 1)
    oled.text(status_line, 0, 12, 1)
    oled.text(f"Sense: {temp_val}C", 0, 22, 1)
    oled.show()

# ------------------------------------------------------------------------------
# 3. NETWORK LOGIC
# ------------------------------------------------------------------------------
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        if HAS_SCREEN:
            oled.fill(0); oled.text("WiFi Connect...", 0, 0, 1); oled.show()
        wlan.connect(WIFI_SSID)
        timeout = 10
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
            
    if wlan.isconnected():
        print('[WiFi] IP:', wlan.ifconfig()[0])
        return True
    return False

def fetch_logs_with_cursor(current_since):
    """
    GET /logs?device_id=...&since=...&limit=...
    """
    # Build URL manually
    url = f"{DB_BASE_URL}?device_id={DEVICE_ID}&limit={LOG_LIMIT}&since={current_since}"
    
    # Print simplified timestamp for debugging
    print(f"[GET] Since: ...{current_since[-10:]}") 
    
    try:
        response = urequests.get(url)
        if response.status_code == 200:
            data = response.json()
            response.close()
            
            # Response Shape: { "logs": [...], "next_since": "..." }
            logs = data.get("logs", [])
            next_since = data.get("next_since", current_since)
            
            return logs, next_since
        else:
            print(f"[GET] Error: {response.status_code}")
            response.close()
            return [], current_since
            
    except Exception as e:
        print(f"[GET] Failed: {e}")
        return [], current_since

def send_to_llm(log_record, real_temp):
    """
    Maps Log fields -> LLM Payload
    """
    print(f"[POST] Sending to LLM...")
    
    # Extract fields from the dataset
    # Fields: timestamp, device_id, level, event_type, error_code, message
    
    sim_error_code = log_record.get("error_code") # Might be None/Null
    if sim_error_code is None:
        sim_error_code = "None"
        
    payload = {
        # Data from Database (Simulation)
        "fda_error_code": sim_error_code,
        "log_message": log_record.get("message", ""),
        "log_level": log_record.get("level", "INFO"),
        "event_type": log_record.get("event_type", "UNKNOWN"),
        "log_timestamp": log_record.get("timestamp", ""),
        
        # Data from Physical World (ESP32)
        "sensor_temp": real_temp,
        
        # Metadata
        "device_id": DEVICE_ID,
        "sender": "ESP32_Bridge"
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = urequests.post(LLM_URL, json=payload, headers=headers)
        if response.status_code == 200:
            update_display("Sent to LLM!", real_temp)
            print("   -> LLM Success")
        else:
            print(f"   -> LLM Fail: {response.status_code}")
        response.close()
        return True
    except Exception as e:
        print(f"[POST] Error: {e}")
        return False

# ------------------------------------------------------------------------------
# 4. MAIN LOOP
# ------------------------------------------------------------------------------
def main():
    if not connect_wifi(): return

    # INITIAL CURSOR
    # Start in the past to ensure we get data. 
    # Use current ISO time if you ONLY want live data.
    cursor_since = "2024-01-01T00:00:00.000Z"

    print("System Online. Monitoring Logs...")
    
    while True:
        current_temp = read_sensor_temperature()
        update_display("Scanning DB...", current_temp)
        
        # 1. Fetch Logs
        logs, next_since = fetch_logs_with_cursor(cursor_since)
        
        # 2. Update Cursor (Always move forward)
        if next_since and next_since != "":
            cursor_since = next_since

        # 3. Process Logic
        if len(logs) > 0:
            print(f"   -> Processing {len(logs)} logs...")
            update_display("Sending Log...", current_temp)
            
            # Since limit=1, we take the first
            latest_log = logs[0]
            
            # 4. Send to LLM
            send_to_llm(latest_log, current_temp)
        else:
            print("   -> No new logs.")
            
        time.sleep(POLL_DELAY)

if __name__ == "__main__":
    main()