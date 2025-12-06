from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn
import fastapi_poe as fp
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import re

app = FastAPI(title="Medical Device Diagnostic - RAG + Few-Shot")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
POE_API_KEY = "aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w"
LLM_MODEL = "GPT-4o-Mini"
TRAINING_DATA_PATH = "medical_device_training.jsonl"

# ==============================================================================
# DATA MODELS
# ==============================================================================

class TemperatureEntry(BaseModel):
    ts: float
    val: float
    humidity: Optional[float] = None

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
# BIOBERT SETUP (RAG)
# ==============================================================================

print("Loading BioBERT for RAG...")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
biobert_model = AutoModel.from_pretrained(MODEL_NAME).to('cuda' if torch.cuda.is_available() else 'cpu')
biobert_model.eval()
print(f"BioBERT loaded on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Load training data from JSONL
print(f"Loading training data from {TRAINING_DATA_PATH}...")
training_data = []
try:
    with open(TRAINING_DATA_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            training_data.append(data)
    print(f"Loaded {len(training_data)} training samples")
    HAS_TRAINING_DATA = True
except Exception as e:
    print(f"Warning: Could not load training data: {e}")
    training_data = []
    HAS_TRAINING_DATA = False

def extract_error_info(user_message):
    """Extract key info from user message for embedding"""
    # Extract error code
    error_code_match = re.search(r'Error Code: (ERROR \d+)', user_message)
    error_code = error_code_match.group(1) if error_code_match else "UNKNOWN"
    
    # Extract device type
    device_match = re.search(r'Device ID: (\w+)-training', user_message)
    device_type = device_match.group(1) if device_match else "unknown_device"
    
    # Extract temperature range
    temp_match = re.search(r'Range: ([\d.]+)°C - ([\d.]+)°C', user_message)
    if temp_match:
        temp_min = float(temp_match.group(1))
        temp_max = float(temp_match.group(2))
        temp_avg = (temp_min + temp_max) / 2
    else:
        temp_avg = 25.0
    
    return error_code, device_type, temp_avg

def extract_diagnosis_summary(assistant_message):
    """Extract key diagnosis info from assistant response"""
    # Extract FDA code
    fda_match = re.search(r'FDA Error Code: ([^\n]+)', assistant_message)
    fda_code = fda_match.group(1).strip() if fda_match else "Unknown"
    
    # Extract severity
    severity_match = re.search(r'Severity: (\w+)', assistant_message)
    severity = severity_match.group(1) if severity_match else "Unknown"
    
    # Extract root cause (first line of root cause section)
    cause_match = re.search(r'Primary cause: ([^\n|]+)', assistant_message)
    root_cause = cause_match.group(1).strip() if cause_match else "Unknown"
    
    # Extract troubleshooting (first step)
    troubleshoot_match = re.search(r'1\. ([^\n|]+)', assistant_message)
    troubleshooting = troubleshoot_match.group(1).strip() if troubleshoot_match else "Unknown"
    
    return fda_code, severity, root_cause, troubleshooting

def embed_text(text):
    """Generate BioBERT embedding"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(biobert_model.device)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return vec

# Create training embeddings at startup
TRAINING_EMBEDDINGS = None
TRAINING_INFO = []

if HAS_TRAINING_DATA:
    print("Creating embeddings for training data...")
    training_texts = []
    
    for sample in training_data:
        # Extract user and assistant messages
        messages = sample.get('messages', [])
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
        
        # Extract structured info
        error_code, device_type, temp_avg = extract_error_info(user_msg)
        fda_code, severity, root_cause, troubleshooting = extract_diagnosis_summary(assistant_msg)
        
        # Create embedding text
        text = f"Error: {error_code}, Device: {device_type}, Temp: {temp_avg:.1f}°C"
        training_texts.append(text)
        
        # Store structured info
        TRAINING_INFO.append({
            'error_code': error_code,
            'device_type': device_type,
            'temperature': temp_avg,
            'fda_code': fda_code,
            'severity': severity,
            'root_cause': root_cause,
            'troubleshooting': troubleshooting,
            'full_user_msg': user_msg[:500],  # Store snippet
            'full_assistant_msg': assistant_msg[:500]
        })
    
    embeddings = []
    for text in tqdm(training_texts, desc="Embeddings"):
        embeddings.append(embed_text(text))
    
    TRAINING_EMBEDDINGS = np.vstack(embeddings)
    print(f"Embeddings ready! ({len(TRAINING_INFO)} cases indexed)")

def find_similar_cases(error_log, temps, top_k=3):
    """Find similar historical cases using BioBERT"""
    if not HAS_TRAINING_DATA:
        return []
    
    # Create query text
    avg_temp = sum(t.val for t in temps) / len(temps) if temps else 0
    query_text = f"Error: {error_log.error_code}, Device: medical device, Temp: {avg_temp:.1f}°C"
    
    # Get embedding and find similar
    query_embedding = embed_text(query_text).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, TRAINING_EMBEDDINGS)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_cases = []
    for idx in top_indices:
        info = TRAINING_INFO[idx]
        similar_cases.append({
            "similarity": float(similarities[idx]),
            "error_code": info['error_code'],
            "device_type": info['device_type'],
            "temperature": info['temperature'],
            "severity": info['severity'],
            "fda_code": info['fda_code'],
            "root_cause": info['root_cause'],
            "troubleshooting": info['troubleshooting']
        })
    
    return similar_cases

# ==============================================================================
# LLM FUNCTIONS
# ==============================================================================

async def get_llm_response(prompt):
    """Query Poe API"""
    full_response = []
    try:
        async for partial in fp.get_bot_response(
            messages=[fp.ProtocolMessage(role="user", content=prompt)],
            bot_name=LLM_MODEL,
            api_key=POE_API_KEY
        ):
            full_response.append(partial.text)
        return "".join(full_response)
    except Exception as e:
        print(f"[ERROR] LLM query failed: {e}")
        return None

def extract_from_llm_response(llm_response):
    """Clean markdown code blocks"""
    if "```" in llm_response:
        llm_response = llm_response.split("```")[0].strip()
    return llm_response

# ==============================================================================
# ANALYSIS WITH RAG + FEW-SHOT
# ==============================================================================

async def analyze_with_rag_fewshot(packet: ErrorPacket):
    """Main analysis function using RAG + Few-Shot"""
    
    # Find error log
    error_log = next((log for log in packet.log_sequence if log.error_code), None)
    if not error_log:
        return "No error found in log sequence"
    
    error_index = packet.log_sequence.index(error_log)
    
    # Analyze temperature
    temps = [t.val for t in packet.temperature_data]
    temp_analysis = {
        "min": min(temps) if temps else 0,
        "max": max(temps) if temps else 0,
        "avg": sum(temps) / len(temps) if temps else 0,
        "trend": "rising" if temps and temps[-1] > temps[0] else "falling" if temps and temps[-1] < temps[0] else "stable"
    }
    
    # Analyze humidity
    humidities = [t.humidity for t in packet.temperature_data if t.humidity is not None]
    humidity_analysis = {
        "min": min(humidities) if humidities else 0,
        "max": max(humidities) if humidities else 0,
        "avg": sum(humidities) / len(humidities) if humidities else 0,
        "trend": "rising" if humidities and humidities[-1] > humidities[0] else "falling" if humidities and humidities[-1] < humidities[0] else "stable"
    }
    
    # Find similar cases (RAG)
    similar_cases = find_similar_cases(error_log, packet.temperature_data, top_k=3)
    
    # Build prompt with RAG + Few-Shot
    prompt = f"""You are a medical device diagnostic AI assistant specialized in FDA-regulated equipment.

╔═══════════════════════════════════════════════════════════════════╗
║                    FEW-SHOT EXAMPLES                               ║
╚═══════════════════════════════════════════════════════════════════╝

EXAMPLE 1: Temperature Sensor Failure
─────────────────────────────────────
Error Code: E-SENSOR-FAIL
Device: Ventilator
Temp Readings: [21.2, 21.0, -999, -999] (sensor disconnect)
Logs: [INFO] System OK → [WARN] Sensor fluctuation → [ERROR] Sensor failure

Diagnosis:
- FDA Error Code: Sensor Malfunction (SMF-001)
- Severity: High
- Root Cause: Temperature sensor hardware failure (reading -999 indicates disconnect)
- Troubleshooting:
  1. Check sensor cable connections
  2. Replace temperature sensor module
  3. Recalibrate after replacement

─────────────────────────────────────
EXAMPLE 2: Thermal Management Failure
─────────────────────────────────────
Error Code: E-TEMP-OVER-THRESH
Device: Medical Monitor
Temp Readings: [72.1, 73.4, 75.8, 78.2] (rising trend)
Logs: [INFO] Normal → [WARN] Fan RPM low → [WARN] Temp rising → [ERROR] Threshold exceeded

Diagnosis:
- FDA Error Code: Thermal Management Failure (TMF-003)
- Severity: Critical
- Root Cause: Cooling system degradation
- Troubleshooting:
  1. IMMEDIATE: Power down device
  2. Inspect cooling fan for obstructions
  3. Check ventilation ports
  4. Replace cooling fan if RPM <2000

─────────────────────────────────────
EXAMPLE 3: Normal Variation
─────────────────────────────────────
Error Code: E-TEMP-VARIANCE
Device: Imaging Device
Temp Readings: [70.8, 71.2, 70.5, 71.0] (stable)
Logs: [INFO] All systems nominal → [INFO] Regular operation

Diagnosis:
- FDA Error Code: Normal Variation (NV-000)
- Severity: Low
- Root Cause: Normal thermal fluctuation within acceptable range
- Troubleshooting:
  1. Continue monitoring
  2. No immediate action required

"""

    # Add RAG context if available
    if similar_cases:
        prompt += f"""
╔═══════════════════════════════════════════════════════════════════╗
║              SIMILAR HISTORICAL CASES (RAG)                        ║
╚═══════════════════════════════════════════════════════════════════╝

"""
        for i, case in enumerate(similar_cases, 1):
            prompt += f"""Case {i} (Similarity: {case['similarity']:.1%}):
- Error: {case['error_code']} | Device: {case['device_type']}
- Temperature: {case['temperature']:.1f}°C | Severity: {case['severity']}
- FDA Code: {case['fda_code']}
- Root Cause: {case['root_cause']}
- Troubleshooting: {case['troubleshooting']}

"""

    # Add current case
    prompt += f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    CURRENT CASE TO ANALYZE                         ║
╚═══════════════════════════════════════════════════════════════════╝

**Device ID:** {packet.device_id}
**Error Code:** {error_log.error_code}
**Error Message:** {error_log.message}
**Error Time:** {error_log.timestamp}

**Temperature Analysis:**
- Range: {temp_analysis['min']:.1f}°C - {temp_analysis['max']:.1f}°C
- Average: {temp_analysis['avg']:.1f}°C
- Trend: {temp_analysis['trend'].upper()}
- Samples: {len(temps)}

**Humidity Analysis:**
- Range: {humidity_analysis['min']:.1f}% - {humidity_analysis['max']:.1f}%
- Average: {humidity_analysis['avg']:.1f}%
- Trend: {humidity_analysis['trend'].upper()}

**Logs Leading to Error:**
"""
    
    # Add logs before error
    start_idx = max(0, error_index - 5)
    for i in range(start_idx, error_index):
        log = packet.log_sequence[i]
        prompt += f"  [{log.level}] {log.message}\n"
    
    prompt += f"\n**>>> ERROR <<<**\n  [{error_log.level}] {error_log.message}\n\n"
    
    # Add logs after error
    prompt += "**Logs After Error:**\n"
    end_idx = min(len(packet.log_sequence), error_index + 6)
    for i in range(error_index + 1, end_idx):
        log = packet.log_sequence[i]
        prompt += f"  [{log.level}] {log.message}\n"
    
    prompt += f"""

╔═══════════════════════════════════════════════════════════════════╗
║                    YOUR DIAGNOSIS                                  ║
╚═══════════════════════════════════════════════════════════════════╝

Analyze this case using:
1. The few-shot examples above as templates
2. The similar historical cases (if provided) as context
3. The actual log sequence and temperature data

Provide diagnosis in this format:

## FDA ERROR CLASSIFICATION
- FDA Error Code: [Code]
- Severity: [Level]
- Device Status: [Status]

## ROOT CAUSE ANALYSIS
- Primary Cause: [explanation]
- Contributing Factors: [list]
- Evidence from logs: [cite specific logs]
- Evidence from temperature: [cite trends]
- Confidence: [percentage]

## TROUBLESHOOTING STEPS
1. IMMEDIATE: [action]
2. DIAGNOSTIC: [action]
3. CORRECTIVE: [action]
4. PREVENTIVE: [action]

## SAFETY ASSESSMENT
- Patient Risk: [level]
- Continue Use: [Yes/No/Restrictions]
- Escalate: [conditions]

Base your analysis on the patterns shown in examples and similar cases.
"""

    # Get LLM response
    print("\n[LLM] Querying GPT-4o-mini with RAG + Few-Shot...")
    response = await get_llm_response(prompt)
    
    if response:
        response = extract_from_llm_response(response)
        print(f"[LLM] Analysis complete ({len(response)} characters)")
        return response
    else:
        return "LLM analysis failed"

# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def save_diagnostic_report(packet: ErrorPacket, diagnosis: str, similar_cases: list):
    """Save complete report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_rag_fewshot_{timestamp}.txt"
    
    error_log = next((log for log in packet.log_sequence if log.error_code), None)
    
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MEDICAL DEVICE DIAGNOSTIC REPORT\n")
        f.write("Model: RAG + Few-Shot with GPT-4o-mini + BioBERT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now()}\n")
        f.write(f"Device ID: {packet.device_id}\n\n")
        
        if error_log:
            f.write("ERROR SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Error Code: {error_log.error_code}\n")
            f.write(f"Message: {error_log.message}\n")
            f.write(f"Timestamp: {error_log.timestamp}\n\n")
        
        if similar_cases:
            f.write("SIMILAR HISTORICAL CASES (RAG)\n")
            f.write("-"*70 + "\n")
            for i, case in enumerate(similar_cases, 1):
                f.write(f"{i}. {case['error_code']} (similarity: {case['similarity']:.1%})\n")
                f.write(f"   Device: {case['device_type']}, Temp: {case['temperature']:.1f}°C\n")
                f.write(f"   Severity: {case['severity']}, FDA Code: {case['fda_code']}\n")
                f.write(f"   Root Cause: {case['root_cause']}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("LLM DIAGNOSTIC ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(diagnosis)
        f.write("\n\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    return filename

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/api/ingest")
async def ingest_error_packet(packet: ErrorPacket):
    """Receive error packet and analyze with RAG + Few-Shot"""
    
    print("\n" + "="*70)
    print("RECEIVED ERROR PACKET")
    print("="*70)
    print(f"Device: {packet.device_id}")
    print(f"Logs: {len(packet.log_sequence)}")
    print(f"Temps: {len(packet.temperature_data)}")
    
    error_log = next((log for log in packet.log_sequence if log.error_code), None)
    if error_log:
        print(f"Error: {error_log.error_code} - {error_log.message}")
    
    # Find similar cases
    similar_cases = []
    if error_log:
        similar_cases = find_similar_cases(error_log, packet.temperature_data)
        if similar_cases:
            print(f"\nFound {len(similar_cases)} similar cases:")
            for case in similar_cases:
                print(f"  • {case['error_code']} (similarity: {case['similarity']:.1%})")
    
    # Analyze
    diagnosis = await analyze_with_rag_fewshot(packet)
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    # Save report
    filename = save_diagnostic_report(packet, diagnosis, similar_cases)
    print(f"✓ Report saved: {filename}\n")
    
    return {
        "status": "success",
        "diagnosis": diagnosis,
        "report_file": filename,
        "model": "RAG + Few-Shot (BioBERT + GPT-4o-mini)",
        "similar_cases_used": len(similar_cases),
        "similar_cases": similar_cases,
        "approach": "hybrid_rag_fewshot"
    }

@app.get("/")
def root():
    return {
        "service": "Medical Device Diagnostic - RAG + Few-Shot",
        "model": "BioBERT (RAG) + GPT-4o-mini (Few-Shot)",
        "training_samples": len(training_data) if HAS_TRAINING_DATA else 0,
        "status": "operational"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "approach": "RAG + Few-Shot"}

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Medical Device Diagnostic Server - RAG + Few-Shot")
    print("="*70)
    print(f"\nLLM Model: {LLM_MODEL}")
    print(f"BioBERT Model: {MODEL_NAME}")
    print(f"Training Data: {TRAINING_DATA_PATH}")
    print(f"Training Samples: {len(training_data) if HAS_TRAINING_DATA else 0}")
    print("\nEndpoints:")
    print("  POST /api/ingest - Receive error packets from ESP32")
    print("  GET  /           - Server info")
    print("  GET  /health     - Health check")
    print("\nStarting server on http://0.0.0.0:5001")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5001)