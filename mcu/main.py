from machine import Pin, I2C, ADC
import ssd1306
import time
import network
import ujson
import urequests
import ntptime # REQUIRED: To sync clock with internet

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
WIFI_SSID = "Columbia University"
# WIFI_PASS = "" 

# DATABASE & LLM
DB_BASE_URL = "http://192.168.1.XXX:8000/logs"
DEVICE_ID   = "ventilator-01"
LLM_URL     = "http://192.168.1.YYY:5000/api/ingest"

# RECORDING SETTINGS
PRE_ERROR_LOGS  = 20
POST_ERROR_LOGS = 20
POLL_DELAY      = 1   # Faster polling to catch logs quickly

# ------------------------------------------------------------------------------
# 2. HARDWARE SETUP
# ------------------------------------------------------------------------------
i2c = I2C(sda=Pin(22), scl=Pin(20))
try:
    oled = ssd1306.SSD1306_I2C(128, 32, i2c)
    HAS_SCREEN = True
except:
    HAS_SCREEN = False

temp_adc = ADC(Pin(34))
temp_adc.atten(ADC.ATTN_11DB) 

def read_sensor_temperature():
    try:
        raw_val = temp_adc.read()
        return round(73.0 + ((raw_val % 20) / 10.0), 1)
    except:
        return 73.0

def update_display(line1, line2):
    if not HAS_SCREEN: return
    oled.fill(0)
    oled.text(line1, 0, 0, 1)
    oled.text(line2, 0, 12, 1)
    oled.show()

# ------------------------------------------------------------------------------
# 3. TIME & PARSING HELPERS
# ------------------------------------------------------------------------------
def sync_clock():
    """Syncs ESP32 internal clock with internet time"""
    print("[TIME] Syncing NTP...")
    try:
        ntptime.settime() # Sets internal RTC
        print("[TIME] Synced!", time.localtime())
    except:
        print("[TIME] Sync failed")

def parse_iso_simple(iso_str):
    """
    Rough parsing of ISO string to epoch seconds for comparison.
    Format assumption: '2025-12-03T20:37:00.123Z'
    """
    try:
        # Remove fractional seconds and 'Z'
        clean = iso_str.split('.')[0] # 2025-12-03T20:37:00
        parts = clean.replace('T', '-').replace(':', '-').split('-')
        # parts: [Year, Mon, Day, Hour, Min, Sec]
        t_tuple = (int(parts[0]), int(parts[1]), int(parts[2]), 
                   int(parts[3]), int(parts[4]), int(parts[5]), 0, 0)
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
        wlan.connect(WIFI_SSID)
        timeout = 10
        while not wlan.isconnected() and timeout > 0:
            time.sleep(1)
            timeout -= 1
    return wlan.isconnected()

def fetch_logs(current_since):
    # Fetch 1 at a time to keep logic simple and ordered
    url = f"{DB_BASE_URL}?device_id={DEVICE_ID}&limit=1&since={current_since}"
    try:
        res = urequests.get(url)
        if res.status_code == 200:
            data = res.json()
            res.close()
            return data.get("logs", []), data.get("next_since", current_since)
        res.close()
    except:
        pass
    return [], current_since

def send_package_to_llm(log_sequence, temp_history):
    """
    COMBINES Logs and Temps into one request
    """
    print(f"\n[POST] Packaging {len(log_sequence)} logs & Temp Data...")
    update_display("Packaging...", "Sending to LLM")
    
    # 1. Determine Time Interval from the logs
    first_log_time = parse_iso_simple(log_sequence[0].get('timestamp', ''))
    last_log_time  = parse_iso_simple(log_sequence[-1].get('timestamp', ''))
    
    # 2. Filter Temp History for this interval
    relevant_temps = []
    for t_entry in temp_history:
        # Allow a small buffer (e.g. +/- 2 seconds)
        if t_entry['ts'] >= (first_log_time - 2) and t_entry['ts'] <= (last_log_time + 2):
            relevant_temps.append(t_entry)
            
    print(f"[POST] Found {len(relevant_temps)} temps in interval {first_log_time}-{last_log_time}")

    # 3. Construct Payload
    payload = {
        "context": "Error Event Packet",
        "device_id": DEVICE_ID,
        "log_sequence": log_sequence,      # The 20 before + error + 20 after
        "temperature_data": relevant_temps # The temps matching that timeframe
    }
    
    headers = {'Content-Type': 'application/json'}
    try:
        res = urequests.post(LLM_URL, json=payload, headers=headers)
        print(f"[POST] Status: {res.status_code}")
        res.close()
        return True
    except Exception as e:
        print(f"[POST] Failed: {e}")
        return False

# ------------------------------------------------------------------------------
# 5. MAIN LOOP (STATE MACHINE)
# ------------------------------------------------------------------------------
def main():
    if not connect_wifi(): return
    sync_clock() # Sync time so our temp timestamps match server logs
    
    # BUFFERS
    # We keep temp history for safety (last ~300 entries / 10 mins)
    temp_history = [] 
    
    # We keep logs based on logic
    log_buffer = [] 
    
    # STATE
    # "MONITORING" or "COLLECTING_POST_ERROR"
    state = "MONITORING"
    logs_needed_after_error = 0
    
    cursor_since = "2024-01-01T00:00:00.000Z" 

    print("System Online. Buffering...")

    while True:
        # A. READ TEMP (Add to History)
        # Store tuple: {ts: epoch_seconds, val: 73.0}
        current_temp = read_sensor_temperature()
        temp_history.append({"ts": time.time(), "val": current_temp})
        
        # Keep temp buffer manageable (remove old > 500 items)
        if len(temp_history) > 500:
            temp_history.pop(0)

        # B. FETCH LOGS
        new_logs, next_since = fetch_logs(cursor_since)
        if next_since: cursor_since = next_since

        # C. PROCESS LOGS
        for log in new_logs:
            # Add to our buffer
            log_buffer.append(log)
            print(f"[LOG] {log.get('message', '')} [{state}]")

            # LOGIC ENGINE
            if state == "MONITORING":
                # 1. Prune Buffer: We only need to keep 'PRE_ERROR_LOGS' 
                # until we find an error.
                if len(log_buffer) > PRE_ERROR_LOGS + 1: 
                    log_buffer.pop(0) 

                # 2. Check for Trigger
                if log.get("error_code") is not None:
                    print("!!! ERROR DETECTED !!! Switching to Capture Mode.")
                    state = "COLLECTING_POST_ERROR"
                    logs_needed_after_error = POST_ERROR_LOGS
                    update_display("ERROR FOUND!", "Capturing Context")

            elif state == "COLLECTING_POST_ERROR":
                # We just accepted a log, so we need one less
                logs_needed_after_error -= 1
                
                update_display("Capturing...", f"Need: {logs_needed_after_error}")
                
                if logs_needed_after_error <= 0:
                    # WE ARE DONE!
                    # The log_buffer now contains: ~20 pre, 1 error, 20 post
                    send_package_to_llm(log_buffer, temp_history)
                    
                    # Reset
                    log_buffer = [] # Clear buffer
                    state = "MONITORING"
                    update_display("Sent!", "Monitoring...")
        
        time.sleep(POLL_DELAY)

if __name__ == "__main__":
    main()