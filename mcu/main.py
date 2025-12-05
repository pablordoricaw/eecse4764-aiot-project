# ==============================================================================
# JASON'S FINAL SYSTEM (Light Sensor Simulating Temp)
# ------------------------------------------------------------------------------
# 1. Polls Mock Database for Logs (Port 8000)
# 2. Reads Light Sensor as "Temperature" (Pin 34)
# 3. If Database sends ERROR -> Captures context -> Sends to LLM (Port 5001)
# ==============================================================================
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
WIFI_SSID = "richard5iphone"
WIFI_PASS = "Sergeant213" # <--- FILL THIS IN!

# SERVER CONFIGURATION (Using your Hotspot IP)
# Mock Database (Pablo)
DB_BASE_URL = "http://172.20.10.13:8000/logs" 
DEVICE_ID   = "ventilator-01"

# LLM Server (Richard)
LLM_URL     = "http://172.20.10.13:5001/api/ingest"

# RECORDING SETTINGS
PRE_ERROR_LOGS  = 20
POST_ERROR_LOGS = 20
POLL_DELAY      = 1.5   # Seconds between checks

# ------------------------------------------------------------------------------
# 2. HARDWARE SETUP (OLED + LIGHT SENSOR)
# ------------------------------------------------------------------------------
# OLED
i2c = I2C(sda=Pin(22), scl=Pin(20))
try:
    oled = ssd1306.SSD1306_I2C(128, 32, i2c)
    HAS_SCREEN = True
except:
    HAS_SCREEN = False

# LIGHT SENSOR (ALS-PT19) on Pin 34
# We use this to simulate temperature.
# Dark = 20C, Bright Light = 100C
light_sensor = ADC(Pin(32))
light_sensor.atten(ADC.ATTN_11DB) # Full Range: 0 - 3.3V

def read_simulated_temperature():
    """
    Converts Light Intensity to Temperature.
    0 (Dark)    -> ~20.0 C
    4095 (Bright)-> ~100.0 C
    """
    try:
        raw_val = light_sensor.read() # 0 to 4095
        
        # Mapping Formula: Temp = Base + (Reading / Max) * Range
        # 20C + (Value / 4095) * 80C
        sim_temp = 20.0 + (raw_val / 4095.0) * 80.0
        
        return round(sim_temp, 1)
    except:
        return 20.0

def update_display(line1, line2):
    if not HAS_SCREEN: return
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
        clean = iso_str.split('.')[0]
        parts = clean.replace('T', '-').replace(':', '-').split('-')
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
    first_ts = parse_iso_simple(log_sequence[0].get('timestamp', ''))
    last_ts  = parse_iso_simple(log_sequence[-1].get('timestamp', ''))
    
    relevant_temps = []
    for t in temp_history:
        # +/- 5 seconds buffer
        if t['ts'] >= (first_ts - 5) and t['ts'] <= (last_ts + 5):
            relevant_temps.append(t)
            
    payload = {
        "context": "Simulated Overheat Event",
        "device_id": DEVICE_ID,
        "log_sequence": log_sequence,
        "temperature_data": relevant_temps
    }
    
    headers = {'Content-Type': 'application/json'}
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
    if not connect_wifi(): return
    sync_clock()
    
    temp_history = [] 
    log_buffer = [] 
    
    state = "MONITORING"
    logs_needed_after_error = 0
    
    # Start looking for logs from roughly "now" (or a fixed past date to be safe)
    cursor_since = "2024-01-01T00:00:00.000Z" 

    print("System Online. Ready for Light/Temp Simulation.")

    while True:
        # A. READ SENSOR (Light -> Temp)
        current_temp = read_simulated_temperature()
        temp_history.append({"ts": time.time(), "val": current_temp})
        if len(temp_history) > 300: temp_history.pop(0)

        # B. FETCH LOGS (From Mock DB)
        new_logs, next_since = fetch_logs(cursor_since)
        if next_since: cursor_since = next_since

        # C. PROCESS LOGS
        for log in new_logs:
            log_buffer.append(log)
            print(f"[LOG] {log.get('message', '')}")

            if state == "MONITORING":
                # Keep buffer small
                if len(log_buffer) > PRE_ERROR_LOGS + 1: log_buffer.pop(0) 

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
            update_display("Monitoring...", f"Temp: {current_temp}C")
            
        time.sleep(POLL_DELAY)

if __name__ == "__main__":
    main()