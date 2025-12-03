from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fastapi_poe as fp
import asyncio
from datetime import datetime

app = FastAPI()

POE_API_KEY = "aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w"

SYSTEM_PROMPT = """You are a medical device diagnostic assistant specialized in analyzing FDA-regulated medical equipment errors and sensor data. Your role is to classify errors, detect anomalies, and generate actionable diagnostic reports.

### Input Data Format
You will receive:
1. FDA error code (e.g., ERROR 2, ERROR 3)
2. Device type and model (e.g., rhinolaryngoscope)
3. Device characteristics and sensor readings:
   - Temperature (°C)
   - Operating parameters
   - Configuration status
4. Timestamp and context information

### Your Tasks
1. **Error Classification**: Map internal error codes to standardized FDA error classifications
2. **Anomaly Detection**: Identify unusual patterns in sensor data or device behavior
3. **Root Cause Analysis**: Determine the most likely cause of the error based on available data
4. **Risk Assessment**: Evaluate patient safety implications

### Output Format
Generate a structured report containing:

**FDA Error Code**: [Standardized classification]
**Severity Level**: [Critical/High/Medium/Low]
**Device Status**: [Operational/Degraded/Failed/Needs Maintenance]

**Diagnostic Hypothesis**:
- Primary cause: [explanation]
- Contributing factors: [if any]
- Confidence level: [percentage]

**Troubleshooting Steps**:
1. [Immediate action required]
2. [Verification steps]
3. [Corrective measures]
4. [Prevention recommendations]

**Additional Notes**: [Any relevant observations or patterns]

### Examples

**Input**: 
- Error Code: ERROR 3
- Device: Rhinolaryngoscope Model XYZ
- Temperature: 45°C
- Expected range: 20-35°C

**Output**:
FDA Error Code: Thermal Management Failure (TMF-003)
Severity Level: High
Device Status: Degraded - Requires immediate inspection

Diagnostic Hypothesis:
- Primary cause: Cooling system malfunction causing elevated operating temperature
- Contributing factors: Possible blocked ventilation or fan failure
- Confidence level: 85%

Troubleshooting Steps:
1. IMMEDIATE: Power down device and allow cooling period (15 min)
2. Inspect ventilation ports for obstructions
3. Test cooling fan operation (should run at 3000 RPM ±5%)
4. Check thermal sensor calibration
5. If issue persists, replace cooling assembly per service manual Section 4.2

Additional Notes: Temperature exceeded safe threshold by 10°C. Review recent usage patterns for abnormal duty cycles.
"""

class DeviceData(BaseModel):
    error_code: str
    device_type: str
    temperature: float
    additional_info: str = ""
    device_id: str = "UNKNOWN"

@app.post("/diagnose")
async def diagnose_device(data: DeviceData):
    """Receive data from ESP32 and return LLM diagnosis"""
    
    print(f"\n📥 Received data from device: {data.device_id}")
    print(f"   Error: {data.error_code}")
    print(f"   Device: {data.device_type}")
    print(f"   Temp: {data.temperature}°C")
    
    user_prompt = f"""Analyze this medical device error:

Error Code: {data.error_code}
Device Type: {data.device_type}
Temperature Reading: {data.temperature}°C
Normal Operating Range: 20-35°C
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Device ID: {data.device_id}
{f"Additional Info: {data.additional_info}" if data.additional_info else ""}
"""

    messages = [
        fp.ProtocolMessage(role="system", content=SYSTEM_PROMPT),
        fp.ProtocolMessage(role="user", content=user_prompt)
    ]
    
    full_response = ""
    
    try:
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name="GPT-4o-mini",
            api_key=POE_API_KEY
        ):
            full_response += partial.text
        
        print(f"Diagnosis generated for device {data.device_id}")
        
        return {
            "status": "success",
            "device_id": data.device_id,
            "diagnosis": full_response,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Medical Device Diagnostic Server Running", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Medical Device Diagnostic Server...")
    print("Listening for ESP32 connections on http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)