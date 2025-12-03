
import urequests
import ujson
import time
from machine import Pin, ADC
import network

# WiFi Configuration
WIFI_SSID = "your_wifi_name"
WIFI_PASSWORD = "your_wifi_password"

# Server Configuration
SERVER_URL = "http://YOUR_COMPUTER_IP:5000/diagnose"  # Replace with your computer's IP
DEVICE_ID = "ESP32_RHINO_001"

# Temperature sensor setup
temp_sensor = ADC(Pin(34))
temp_sensor.atten(ADC.ATTN_11DB)

def connect_wifi():
    """Connect to WiFi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print("Connecting to WiFi...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        
        while not wlan.isconnected():
            time.sleep(1)
            print(".", end="")
    
    print(f"\n✓ Connected! IP: {wlan.ifconfig()[0]}")
    return wlan.ifconfig()[0]

def read_temperature():
    """Read temperature from ADC sensor"""
    adc_value = temp_sensor.read()
    voltage = (adc_value / 4095.0) * 3.3
    temp_c = (voltage - 0.5) * 100  # For TMP36 sensor
    return round(temp_c, 1)

def simulate_error(temp):
    """Determine error code based on temperature"""
    if temp > 40:
        return "ERROR 3", "Temperature exceeds safe operating threshold"
    elif temp < 15:
        return "ERROR 3", "Temperature below minimum operating range"
    elif 35 < temp <= 40:
        return "ERROR 2", "Temperature approaching warning level"
    else:
        return None, "Normal operation"

def send_to_server(error_code, temp, device_type="rhinolaryngoscope", additional_info=""):
    """Send device data to diagnostic server"""
    
    payload = {
        "error_code": error_code,
        "device_type": device_type,
        "temperature": temp,
        "additional_info": additional_info,
        "device_id": DEVICE_ID
    }
    
    try:
        print(f"\n📤 Sending: {error_code} | Temp: {temp}°C")
        
        response = urequests.post(
            SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received:")
            print(f"   Status: {result['status']}")
            print(f"   Diagnosis: {result['diagnosis'][:100]}...")
        else:
            print(f"❌ Server error: {response.status_code}")
            print(f"   Response: {response.text}")
        
        response.close()
        return True
        
    except Exception as e:
        print(f"❌ Error sending data: {e}")
        return False

def main():
    """Main monitoring loop"""
    print("="*50)
    print("Medical Device Monitor - ESP32")
    print("="*50)
    
    # Connect to WiFi
    connect_wifi()
    
    print(f"\n🏥 Monitoring device: {DEVICE_ID}")
    print(f"📡 Server: {SERVER_URL}")
    print("\nStarting continuous monitoring...\n")
    
    while True:
        # Read sensor
        temp = read_temperature()
        
        # Check for errors
        error_code, info = simulate_error(temp)
        
        if error_code:
            print(f"\n⚠️  ALERT: {error_code} detected!")
            print(f"   Temperature: {temp}°C")
            print(f"   Info: {info}")
            
            # Send to server for diagnosis
            send_to_server(error_code, temp, additional_info=info)
        else:
            print(f"✓ Normal - Temp: {temp}°C - {info}")
        
        # Wait before next check
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
