# ==============================================================================
# IMPORTS
# ==============================================================================
from machine import Pin, I2C, ADC
import ssd1306
import time
import network
import ujson
import socket

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# WiFi Configuration
WIFI_SSID = "Columbia University"
# WIFI_PASSWORD = "00001111"

# Server Configuration
SERVER_PORT = 8080

# ==============================================================================
# HARDWARE INITIALIZATION
# ==============================================================================
# Temperature sensor (using ADC pin for simulation)
temp_adc = ADC(Pin(34))
temp_adc.atten(ADC.ATTN_11DB)  # 0-3.3V range

# I2C and Display
oled_width = 128
oled_height = 32
i2c = I2C(sda=Pin(22), scl=Pin(20))
oled = ssd1306.SSD1306_I2C(oled_width, oled_height, i2c)

# ==============================================================================
# GLOBAL STATE VARIABLES
# ==============================================================================
wifi_connected = False
TEMP_SENSOR_AVAILABLE = True
current_temp = 0.0

# ==============================================================================
# TEMPERATURE SENSOR FUNCTIONS
# ==============================================================================
def read_sensor_temperature():
    """
    Read actual temperature from sensor
    For demo: simulates temperature around 73°C with some variance
    Replace with actual sensor reading logic (DS18B20, DHT22, etc.)
    """
    try:
        # Read ADC value
        adc_value = temp_adc.read()
        
        # Simulate temperature reading around 73°C ± 5°C
        # In real implementation, use proper sensor calibration
        base_temp = 73.0
        variance = (adc_value % 100) / 10.0 - 5.0  # ±5°C variance
        temp_celsius = base_temp + variance
        
        print(f"[SENSOR] Temperature reading: {temp_celsius:.1f}°C")
        return round(temp_celsius, 1)
        
    except Exception as e:
        print(f"[SENSOR] Error reading temperature: {e}")
        return 73.0  # Default fallback value

# ==============================================================================
# STAGE 1 DATA COLLECTION (EMPTY - TO BE IMPLEMENTED)
# ==============================================================================
def receive_stage1_data():
    """
    Receive data from Stage 1 (Simulated Medical Device)
    
    TODO: Implement communication protocol with Stage 1
    Expected data format:
    {
        "error_code": "ERROR_3",
        "error_type": "TEMP",
        "simulated_device_temp": 100.0,
        "device_info": {
            "device_type": "rhinolaryngoscope",
            "device_model": "...",
            ...
        }
    }
    """
    # EMPTY - TO BE IMPLEMENTED
    # This is where you'll receive data from Pablo/Rahul's simulated medical device
    pass

# ==============================================================================
# WIFI FUNCTIONS
# ==============================================================================
def connect_wifi():
    """Connect to WiFi network"""
    global wifi_connected
    
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"[WiFi] Connecting to {WIFI_SSID}...")
        wlan.connect(WIFI_SSID)
        
        timeout = 0
        while not wlan.isconnected() and timeout < 20:
            time.sleep(1)
            timeout += 1
            print(".", end="")
        
        print()
        
        if wlan.isconnected():
            wifi_connected = True
            print(f"[WiFi] Connected! IP: {wlan.ifconfig()[0]}")
        else:
            wifi_connected = False
            print(f"[WiFi] Failed to connect to '{WIFI_SSID}'")
    else:
        wifi_connected = True
        print(f"[WiFi] Already connected. IP: {wlan.ifconfig()[0]}")
    
    return wifi_connected

# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================
def show_message(message, line=0):
    """Display a message on OLED at specified line"""
    oled.fill(0)
    oled.text(message, 0, line * 10, 1)
    oled.show()

def display_current_temperature():
    """Display current temperature from sensor on OLED"""
    global current_temp
    current_temp = read_sensor_temperature()
    
    oled.fill(0)
    oled.text("Stage 2 - Temp", 0, 0, 1)
    oled.text(f"Sensor: {current_temp}C", 0, 12, 1)
    oled.show()

def display_report(report):
    """Display LLM diagnostic report on OLED"""
    oled.fill(0)
    oled.text("Report:", 0, 0, 1)
    
    # Display first two lines
    if len(report) <= 16:
        oled.text(report, 0, 12, 1)
    else:
        oled.text(report[:16], 0, 12, 1)
        oled.text(report[16:32], 0, 22, 1)
    
    oled.show()
    print(f"[REPORT] {report}")

# ==============================================================================
# JSON COMMAND HANDLER
# ==============================================================================
def execute_command(command):
    """
    Execute a command based on JSON input
    Expected format: {name: <function_name>, args: [XX, XX]}
    Returns: (status_code: int, success: bool, message: str)
    """
    try:
        if not isinstance(command, dict):
            return 400, False, "Error: Command must be a JSON object"
        
        if "name" not in command:
            return 400, False, "Error: Missing 'name' field in command"
        
        if "args" not in command:
            return 400, False, "Error: Missing 'args' field in command"
        
        func_name = command["name"]
        args = command["args"]
        
        print(f"[CMD] Executing: {func_name}")
        
        if func_name == "collect_error_data":
            # Collect error data and format for LLM
            if not TEMP_SENSOR_AVAILABLE:
                return 503, False, "Error: Temperature sensor not available"
            
            try:
                show_message("Collecting...", 0)
                
                # Read current sensor temperature
                sensor_temp = read_sensor_temperature()
                
                # TODO: Get data from Stage 1
                # stage1_data = receive_stage1_data()
                # For now, create placeholder structure
                
                error_data = {
                    "actual_sensor_temp": sensor_temp,
                    "timestamp": time.time(),
                    "device_info": {
                        "device_type": "Medical_Device",
                        "location": "Lab_Test_Environment",
                        "operator": "Jason"
                    },
                    # Stage 1 data will be added here when implemented
                    "stage1_data": None  # PLACEHOLDER - will come from receive_stage1_data()
                }
                
                # Format for LLM
                llm_payload = {
                    "stage": "stage_2_to_stage_3",
                    "error_data": error_data,
                    "analysis_request": {
                        "task": "medical_device_diagnostic",
                        "required_outputs": [
                            "FDA_error_code_interpretation",
                            "troubleshooting_steps",
                            "device_status_assessment",
                            "safety_recommendations"
                        ]
                    }
                }
                
                data_str = ujson.dumps(llm_payload)
                
                # Display current temperature
                display_current_temperature()
                
                print(f"[CMD] Error data collected: {len(data_str)} bytes")
                return 200, True, data_str
                
            except Exception as e:
                return 500, False, f"Error collecting data: {str(e)}"
        
        elif func_name == "display_report":
            # Display LLM diagnostic report
            if len(args) < 1:
                return 400, False, "Error: display_report requires 1 argument (report)"
            
            report = str(args[0])
            display_report(report)
            return 200, True, "Report displayed successfully"
        
        elif func_name == "get_temperature":
            # Get current temperature reading
            sensor_temp = read_sensor_temperature()
            display_current_temperature()
            
            status = {
                "sensor_temp": sensor_temp,
                "timestamp": time.time()
            }
            
            return 200, True, ujson.dumps(status)
        
        else:
            return 400, False, f"Error: Unknown function '{func_name}'"
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 500, False, f"Internal Server Error: {str(e)}"

# ==============================================================================
# HTTP SERVER
# ==============================================================================
def parse_http_request(request):
    """Parse HTTP request and extract JSON body"""
    try:
        parts = request.split(b"\r\n\r\n", 1)
        if len(parts) < 2:
            return None
        
        body = parts[1].decode("utf-8").strip()
        if not body:
            return None
        
        return ujson.loads(body)
    except Exception as e:
        print(f"[SERVER] Error parsing request: {e}")
        return None

def create_http_response(status_code, success, message):
    """Create HTTP response with JSON body"""
    response_data = {"success": success, "message": message}
    body = ujson.dumps(response_data)
    
    status_messages = {
        200: "OK",
        400: "Bad Request",
        500: "Internal Server Error",
        503: "Service Unavailable",
    }
    
    status_text = status_messages.get(status_code, "Unknown")
    
    response = f"HTTP/1.1 {status_code} {status_text}\r\n"
    response += "Content-Type: application/json\r\n"
    response += f"Content-Length: {len(body)}\r\n"
    response += "Connection: close\r\n"
    response += "\r\n"
    response += body
    
    return response.encode("utf-8")

def start_server():
    """Start HTTP server to listen for commands"""
    if not wifi_connected:
        print("[SERVER] WiFi not connected. Cannot start server.")
        return
    
    addr = socket.getaddrinfo("0.0.0.0", SERVER_PORT)[0][-1]
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(addr)
    s.listen(1)
    
    print(f"[SERVER] Listening on port {SERVER_PORT}...")
    print(
        f"[SERVER] Connect to: http://{network.WLAN(network.STA_IF).ifconfig()[0]}:{SERVER_PORT}"
    )
    
    while True:
        try:
            cl, addr = s.accept()
            print(f"[SERVER] Connection from {addr}")
            
            cl.settimeout(10.0)
            request = cl.recv(1024)
            
            command = parse_http_request(request)
            
            if command:
                print(f"[SERVER] Received command: {command}")
                status_code, success, message = execute_command(command)
                print(f"[SERVER] Result ({status_code}): {message[:100]}...")
            else:
                status_code = 400
                success = False
                message = "Error: Invalid or missing JSON command"
            
            response = create_http_response(status_code, success, message)
            cl.send(response)
            cl.close()
        
        except Exception as e:
            print(f"[SERVER] Error handling request: {e}")
            try:
                cl.close()
            except:
                pass

# ==============================================================================
# BACKGROUND TEMPERATURE DISPLAY UPDATE
# ==============================================================================
def update_temperature_display():
    """Periodically update temperature display"""
    display_current_temperature()

# ==============================================================================
# MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Stage 2: Medical Device Diagnostic System")
    print("Error Reading & Sensor Data Collection")
    print("=" * 50 + "\n")
    
    # Show startup message on OLED
    show_message("Stage 2", 0)
    show_message("Starting...", 1)
    time.sleep(1)
    
    # Connect to WiFi
    if not connect_wifi():
        print("[FATAL] Cannot start server without WiFi connection")
        show_message("WiFi Failed!", 0)
    else:
        # Show ready message
        wlan = network.WLAN(network.STA_IF)
        ip = wlan.ifconfig()[0]
        
        oled.fill(0)
        oled.text("Stage 2 Ready", 0, 0, 1)
        oled.text(f"IP:{ip}", 0, 12, 1)
        oled.text(f"Port:{SERVER_PORT}", 0, 22, 1)
        oled.show()
        
        print("\n[READY] Stage 2 system online")
        print(f"[READY] API endpoint: http://{ip}:{SERVER_PORT}")
        print(f"[INFO] Temperature sensor initialized")
        print(f"[INFO] Current Sensor Temp: {read_sensor_temperature()}°C\n")
        print("[TODO] Stage 1 data reception not yet implemented (receive_stage1_data)\n")
        
        # Display initial temperature
        time.sleep(1)
        display_current_temperature()
        
        # Start server (runs forever)
        start_server()