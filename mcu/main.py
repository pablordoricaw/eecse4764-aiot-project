# ==============================================================================
# JASON'S FINAL SYSTEM (With Anti-Crash I2C Recovery)
# ==============================================================================
from machine import Pin, I2C
import ssd1306
import time
import network
import ujson
import urequests
import ntptime 

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
WIFI_SSID = "richard5iphone"
WIFI_PASS = "Sergeant213" 

DB_BASE_URL = "http://172.20.10.13:8000/logs" 
DEVICE_ID   = "ventilator-01"
LLM_URL     = "http://172.20.10.13:5001/api/ingest"

PRE_ERROR_LOGS  = 20
POST_ERROR_LOGS = 20
POLL_DELAY      = 1.0

# ------------------------------------------------------------------------------
# 2. HARDWARE DRIVERS
# ------------------------------------------------------------------------------
class Si7021:
    _ADDR = 0x40 
    _CMD_MEASURE_TEMP_NO_HOLD = b'\xF3'
    
    def __init__(self, i2c_bus):
        self.i2c = i2c_bus
    
    def read_temp(self):
        try:
            self.i2c.writeto(self._ADDR, self._CMD_MEASURE_TEMP_NO_HOLD)
            time.sleep_ms(25)
            data = self.i2c.readfrom(self._ADDR, 2)
            temp_code = (data[0] << 8) | data[1]
            temp_c = (175.72 * temp_code / 65536.0) - 46.85
            return round(temp_c, 2)
        except:
            return None

# ------------------------------------------------------------------------------
# 3. ROBUST INITIALIZATION (Prevents Boot Loops)
# ------------------------------------------------------------------------------
i2c = None
oled = None
sensor = None
HAS_SCREEN = False

def recover_i2c_bus(scl_pin, sda_pin):
    """
    Manually toggle SCL to unstick a frozen I2C bus.
    This prevents WDT resets on boot if a sensor is hanging.
    """
    print("[HW] Attempting I2C Bus Recovery...")
    scl = Pin(scl_pin, Pin.OUT)
    sda = Pin(sda_pin, Pin.IN)
    
    # Cycle clock 9 times to flush any stuck bits
    for _ in range(9):
        scl.value(0)
        time.sleep_us(10)
        scl.value(1)
        time.sleep_us(10)
    
    # Generate Stop condition
    scl.value(0)
    time.sleep_us(10)
    sda_out = Pin(sda_pin, Pin.OUT)
    sda_out.value(0)
    time.sleep_us(10)
    scl.value(1)
    time.sleep_us(10)
    sda_out.value(1)
    print("[HW] Bus Recovery Complete.")

# ATTEMPT INIT
try:
    # 1. Recover bus first (SCL=22, SDA=23)
    recover_i2c_bus(22, 23)
    
    # 2. Now Initialize I2C safely
    i2c = I2C(sda=Pin(23), scl=Pin(22), freq=100000) # Lower freq for stability
    print("[HW] I2C Bus Initialized")
    
    # 3. Scan bus to see what's actually there
    devices = i2c.scan()
    print(f"[HW] Devices found: {[hex(x) for x in devices]}")
    
    if 0x40 in devices:
        sensor = Si7021(i2c)
        print("[HW] Si7021 Sensor Connected")
    
    if 0x3C in devices:
        oled = ssd1306.SSD1306_I2C(128, 32, i2c)
        HAS_SCREEN = True
        print("[HW] OLED Connected")
        
except Exception as e:
    print(f"[HW] CRITICAL INIT ERROR: {e}")
    # Don't crash, just run without sensors
    pass

# ------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def read_temperature():
    if sensor is None: return -999.0
    t = sensor.read_temp()
    if t is None: return -999.0
    return t

def update_display(line1, line2):
    if not HAS_SCREEN: return
    try:
        oled.fill(0)
        oled.text(line1, 0, 0, 1)
        oled.text(line2, 0, 12, 1)
        oled.show()
    except:
        pass

def sync_clock():
    print("[TIME] Syncing NTP...")
    try:
        ntptime.settime() 
        print("[TIME] Synced!")
    except:
        print("[TIME] Failed")

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
# 5. NETWORK LOGIC
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
    print(f"\n[POST] Sending {len(log_sequence)} logs to LLM...")
    update_display("Packaging...", "Sending to LLM")
    
    first_ts = parse_iso_simple(log_sequence[0].get('timestamp', ''))
    last_ts  = parse_iso_simple(log_sequence[-1].get('timestamp', ''))
    
    relevant_temps = []
    for t in temp_history:
        if t['ts'] >= (first_ts - 5) and t['ts'] <= (last_ts + 5):
            relevant_temps.append(t)
            
    payload = {
        "context": "Si7021 Real Sensor Event",
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
# 6. MAIN LOOP
# ------------------------------------------------------------------------------
def main():
    if not connect_wifi(): return
    sync_clock()
    
    temp_history = [] 
    log_buffer = [] 
    state = "MONITORING"
    logs_needed_after_error = 0
    cursor_since = "2024-01-01T00:00:00.000Z" 

    print("System Online. Reading Si7021...")
    update_display("Monitoring...", "Sensor Ready")

    while True:
        # A. READ SENSOR
        current_temp = read_temperature()
        temp_history.append({"ts": time.time(), "val": current_temp})
        if len(temp_history) > 300: temp_history.pop(0)

        # B. FETCH LOGS
        new_logs, next_since = fetch_logs(cursor_since)
        if next_since: cursor_since = next_since

        # C. PROCESS LOGS
        for log in new_logs:
            log_buffer.append(log)
            print(f"[LOG] {log.get('message', '')}")

            if state == "MONITORING":
                if len(log_buffer) > PRE_ERROR_LOGS + 1: log_buffer.pop(0) 

                if log.get("error_code") is not None:
                    print("!!! ERROR DETECTED !!!")
                    state = "COLLECTING_POST_ERROR"
                    logs_needed_after_error = POST_ERROR_LOGS
                    update_display("ERROR!", "Capturing...")

            elif state == "COLLECTING_POST_ERROR":
                logs_needed_after_error -= 1
                update_display("Capturing...", f"Left: {logs_needed_after_error}")
                
                if logs_needed_after_error <= 0:
                    send_package_to_llm(log_buffer, temp_history)
                    log_buffer = [] 
                    state = "MONITORING"
                    update_display("Sent!", "Monitoring...")
        
        # Display Status
        if state == "MONITORING":
            if current_temp == -999.0:
                # print(f"[WARN] Sensor Disconnected") 
                update_display("Sensor Error", "Check Wires")
            else:
                update_display("Monitoring...", f"Temp: {current_temp}C")
            
        time.sleep(POLL_DELAY)

if __name__ == "__main__":
    main()