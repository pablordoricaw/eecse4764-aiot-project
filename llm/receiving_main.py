from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import fastapi_poe as fp
from datetime import datetime

app = FastAPI(title="Medical Device LLM Analysis Server")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
POE_API_KEY = "aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w"
LLM_MODEL = "GPT-4o-Mini"

# ==============================================================================
# DATA MODELS
# ==============================================================================

class TemperatureEntry(BaseModel):
    ts: float  # Unix timestamp
    val: float  # Temperature value

class LogEntry(BaseModel):
    timestamp: str
    device_id: str
    level: str
    event_type: str
    error_code: Optional[str]
    message: str

class ErrorPacket(BaseModel):
    context: str
    device_id: str
    log_sequence: List[LogEntry]
    temperature_data: List[TemperatureEntry]

# ==============================================================================
# LLM FUNCTIONS
# ==============================================================================

async def get_llm_response(prompt, bot_name=LLM_MODEL):
    """Query LLM and return response"""
    full_response = []
    
    try:
        async for partial in fp.get_bot_response(
            messages=[fp.ProtocolMessage(role="user", content=prompt)],
            bot_name=bot_name,
            api_key=POE_API_KEY
        ):
            full_response.append(partial.text)
        
        return "".join(full_response)
    except Exception as e:
        print(f"[ERROR] LLM query failed: {e}")
        return None

def extract_from_llm_response(llm_response):
    """Remove markdown code blocks if present"""
    if "```" in llm_response:
        llm_response = llm_response.split("```")[0].strip()
    return llm_response

# ==============================================================================
# LLM ANALYSIS
# ==============================================================================

async def analyze_with_llm(packet: ErrorPacket):
    """Send error packet to LLM for analysis"""
    
    # Find the error log
    error_log = None
    for log in packet.log_sequence:
        if log.error_code is not None:
            error_log = log
            break
    
    if not error_log:
        return "No error found in log sequence"
    
    # Format temperature data summary
    temp_summary = f"Temperature readings: {len(packet.temperature_data)} samples\n"
    if packet.temperature_data:
        temps = [t.val for t in packet.temperature_data]
        temp_summary += f"  Min: {min(temps):.1f}°C\n"
        temp_summary += f"  Max: {max(temps):.1f}°C\n"
        temp_summary += f"  Avg: {sum(temps)/len(temps):.1f}°C\n"
        
        # Show recent temperature trend
        temp_summary += f"\nRecent temperature trend:\n"
        for t in packet.temperature_data[-5:]:
            temp_summary += f"  {datetime.fromtimestamp(t.ts).strftime('%H:%M:%S')}: {t.val:.1f}°C\n"
    
    # Count logs before and after error
    error_index = packet.log_sequence.index(error_log)
    logs_before = error_index
    logs_after = len(packet.log_sequence) - error_index - 1
    
    # Create comprehensive prompt
    prompt = f"""You are a medical device diagnostic AI assistant specializing in FDA-regulated equipment troubleshooting.

Analyze the following error event from a medical device:

**DEVICE INFORMATION:**
- Device ID: {packet.device_id}
- Context: {packet.context}
- Total logs captured: {len(packet.log_sequence)}

**ERROR EVENT DETAILS:**
- Timestamp: {error_log.timestamp}
- Error Code: {error_log.error_code}
- Severity Level: {error_log.level}
- Event Type: {error_log.event_type}
- Error Message: {error_log.message}

**CONTEXT ANALYSIS:**
- Logs before error: {logs_before}
- Logs after error: {logs_after}

**LOGS LEADING UP TO ERROR (Last 5):**
"""
    
    # Add logs before error
    start_idx = max(0, error_index - 5)
    for i in range(start_idx, error_index):
        log = packet.log_sequence[i]
        prompt += f"  [{log.level}] {log.timestamp} - {log.message}\n"
    
    prompt += f"\n**>>> ERROR LOG <<<**\n  [{error_log.level}] {error_log.timestamp} - {error_log.message}\n\n"
    
    # Add logs after error
    prompt += f"**LOGS FOLLOWING ERROR (Next 5):**\n"
    end_idx = min(len(packet.log_sequence), error_index + 6)
    for i in range(error_index + 1, end_idx):
        log = packet.log_sequence[i]
        prompt += f"  [{log.level}] {log.timestamp} - {log.message}\n"
    
    prompt += f"\n**TEMPERATURE DATA:**\n{temp_summary}\n"
    
    prompt += """
**COMPREHENSIVE DIAGNOSTIC ANALYSIS REQUIRED:**

1. **FDA Error Code Interpretation**
   - Explain the error code in the context of medical device regulations (IEC 62304, FDA 21 CFR Part 820)
   - Classify severity according to FDA risk classification

2. **Root Cause Analysis**
   - Analyze what caused this error based on:
     * Log sequence patterns before the error
     * Temperature data correlation with the error
     * Any anomalies in the context logs
   - Identify potential failure modes

3. **Troubleshooting Steps**
   - Provide 5 specific, actionable steps for the operator
   - Prioritize steps by urgency and likelihood of success
   - Include verification steps after each action

4. **Device Status Assessment**
   - Determine current device safety status: **Safe to Use** / **Unsafe - Stop Operation** / **Caution - Monitor Closely**
   - Justify the assessment based on error severity and context
   - Specify any operational limitations

5. **Safety Recommendations**
   - Immediate actions required for patient/operator safety
   - Preventive measures to avoid recurrence
   - When to escalate to biomedical engineering or service personnel

**FORMAT:** Keep response structured, concise, and actionable. Use clear section headers.
"""
    
    # Query LLM
    try:
        print("\n[LLM] Sending prompt to LLM...")
        response = await get_llm_response(prompt)
        
        if not response:
            return "LLM analysis failed - no response received"
        
        response = extract_from_llm_response(response)
        print(f"[LLM] Analysis complete ({len(response)} characters)")
        
        return response
        
    except Exception as e:
        return f"LLM analysis failed: {str(e)}"

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def save_diagnostic_report(packet: ErrorPacket, diagnosis: str):
    """Save complete diagnostic report to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_report_{timestamp}.txt"
    
    # Find error log
    error_log = next((log for log in packet.log_sequence if log.error_code), None)
    
    try:
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MEDICAL DEVICE DIAGNOSTIC REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device ID: {packet.device_id}\n")
            f.write(f"Context: {packet.context}\n\n")
            
            if error_log:
                f.write("="*70 + "\n")
                f.write("ERROR SUMMARY\n")
                f.write("="*70 + "\n\n")
                f.write(f"Error Code: {error_log.error_code}\n")
                f.write(f"Error Message: {error_log.message}\n")
                f.write(f"Timestamp: {error_log.timestamp}\n")
                f.write(f"Severity: {error_log.level}\n\n")
            
            f.write("="*70 + "\n")
            f.write("LOG SEQUENCE\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total logs: {len(packet.log_sequence)}\n\n")
            for i, log in enumerate(packet.log_sequence, 1):
                marker = " <<< ERROR" if log.error_code else ""
                f.write(f"{i}. [{log.level}] {log.timestamp} - {log.message}{marker}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TEMPERATURE DATA\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total samples: {len(packet.temperature_data)}\n\n")
            if packet.temperature_data:
                temps = [t.val for t in packet.temperature_data]
                f.write(f"Min: {min(temps):.1f}°C\n")
                f.write(f"Max: {max(temps):.1f}°C\n")
                f.write(f"Avg: {sum(temps)/len(temps):.1f}°C\n\n")
                f.write("Sample readings:\n")
                for t in packet.temperature_data[:10]:  # First 10
                    f.write(f"  {datetime.fromtimestamp(t.ts).strftime('%Y-%m-%d %H:%M:%S')}: {t.val:.1f}°C\n")
                if len(packet.temperature_data) > 10:
                    f.write(f"  ... ({len(packet.temperature_data) - 10} more samples)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("LLM DIAGNOSTIC ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(diagnosis)
            f.write("\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        return filename
        
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")
        return None

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.post("/api/ingest")
async def ingest_error_packet(packet: ErrorPacket):
    """
    Receive error packet from ESP32 and analyze it
    """
    print("\n" + "="*70)
    print("RECEIVED ERROR PACKET FROM ESP32")
    print("="*70)
    print(f"Device ID: {packet.device_id}")
    print(f"Context: {packet.context}")
    print(f"Log sequence: {len(packet.log_sequence)} logs")
    print(f"Temperature data: {len(packet.temperature_data)} samples")
    
    # Find and display error
    error_log = next((log for log in packet.log_sequence if log.error_code), None)
    if error_log:
        print(f"\nERROR DETECTED:")
        print(f"  Code: {error_log.error_code}")
        print(f"  Level: {error_log.level}")
        print(f"  Message: {error_log.message}")
        print(f"  Time: {error_log.timestamp}")
    else:
        print("\nWARNING: No error code found in log sequence")
    
    # Analyze with LLM
    print("\nAnalyzing with LLM...")
    diagnosis = await analyze_with_llm(packet)
    
    print("\n" + "="*70)
    print("LLM DIAGNOSTIC REPORT")
    print("="*70)
    print(diagnosis)
    print("="*70 + "\n")
    
    filename = save_diagnostic_report(packet, diagnosis)
    if filename:
        print(f"✓ Report saved: {filename}\n")
    else:
        print("✗ Failed to save report\n")
    
    return {
        "status": "success",
        "diagnosis": diagnosis,
        "report_file": filename,
        "error_code": error_log.error_code if error_log else None
    }

@app.get("/")
def root():
    return {
        "service": "Medical Device LLM Analysis Server (Stage 3)",
        "version": "1.0",
        "model": LLM_MODEL,
        "status": "operational"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for ESP32 to verify server is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Stage 3: Medical Device LLM Analysis Server")
    print("="*70)
    print(f"\nLLM Model: {LLM_MODEL}")
    print(f"API Key: {POE_API_KEY[:20]}...")
    print("\nEndpoints:")
    print("  POST /api/ingest - Receive error packets from ESP32")
    print("  GET  /           - Server info")
    print("  GET  /health     - Health check")
    print("\nStarting server on http://0.0.0.0:5001")
    print("="*70 + "\n")
    
    # CHANGED PORT FROM 5000 TO 5001 TO AVOID CONFLICT
    uvicorn.run(app, host="0.0.0.0", port=5001)