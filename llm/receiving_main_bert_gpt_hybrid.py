from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import uvicorn
import fastapi_poe as fp
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from json import JSONDecodeError
from tqdm import tqdm
import re

app = FastAPI(title="Medical Device Diagnostic - RAG + Few-Shot")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
POE_API_KEY = "aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w"
LLM_MODEL = "GPT-4.1-Mini"
TRAINING_DATA_PATH = "medical_device_training_with_humidity.jsonl"

# ==============================================================================
# DATA MODELS - UPDATED FOR NEW SCHEMA
# ==============================================================================


class SingleLogPacket(BaseModel):
    """New schema - receives one log at a time"""

    log: dict  # Contains all log fields including sensor data
    log_index: int  # 1-5 (or 1-based indexing)
    log_count: int  # Total logs in this episode


# ==============================================================================
# SESSION MANAGEMENT
# ==============================================================================

# In-memory storage for accumulating logs from ESP32
active_sessions: Dict[str, List[dict]] = {}


# ==============================================================================
# BIOBERT SETUP (RAG)
# ==============================================================================

print("Loading BioBERT for RAG...")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
biobert_model = AutoModel.from_pretrained(MODEL_NAME).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
biobert_model.eval()
print(f"BioBERT loaded on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Load training data from JSONL
print(f"Loading training data from {TRAINING_DATA_PATH}...")
training_data = []
try:
    with open(TRAINING_DATA_PATH, "r") as f:
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
    error_code_match = re.search(r"Error Code: (ERROR \d+)", user_message)
    error_code = error_code_match.group(1) if error_code_match else "UNKNOWN"

    device_match = re.search(r"Device ID: (\w+)-training", user_message)
    device_type = device_match.group(1) if device_match else "unknown_device"

    temp_match = re.search(r"Average: ([\d.]+)°C", user_message)
    temp_avg = float(temp_match.group(1)) if temp_match else 25.0

    humidity_match = re.search(r"Average: ([\d.]+)%", user_message)
    humidity_avg = float(humidity_match.group(1)) if humidity_match else 50.0

    return error_code, device_type, temp_avg, humidity_avg


def extract_diagnosis_summary(assistant_message):
    """Extract key diagnosis info from assistant response"""
    fda_match = re.search(r"FDA Error Code: ([^\n]+)", assistant_message)
    fda_code = fda_match.group(1).strip() if fda_match else "Unknown"

    severity_match = re.search(r"Severity: (\w+)", assistant_message)
    severity = severity_match.group(1) if severity_match else "Unknown"

    cause_match = re.search(r"Primary cause: ([^\n|]+)", assistant_message)
    root_cause = cause_match.group(1).strip() if cause_match else "Unknown"

    troubleshoot_match = re.search(r"1\. ([^\n|]+)", assistant_message)
    troubleshooting = (
        troubleshoot_match.group(1).strip() if troubleshoot_match else "Unknown"
    )

    return fda_code, severity, root_cause, troubleshooting


def embed_text(text):
    """Generate BioBERT embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        biobert_model.device
    )
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
        messages = sample.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )

        error_code, device_type, temp_avg, humidity_avg = extract_error_info(user_msg)
        fda_code, severity, root_cause, troubleshooting = extract_diagnosis_summary(
            assistant_msg
        )

        text = (
            f"Error: {error_code}, Device: {device_type}, "
            f"Temp: {temp_avg:.1f}°C, Humidity: {humidity_avg:.1f}%"
        )
        training_texts.append(text)

        TRAINING_INFO.append(
            {
                "error_code": error_code,
                "device_type": device_type,
                "temperature": temp_avg,
                "humidity": humidity_avg,
                "fda_code": fda_code,
                "severity": severity,
                "root_cause": root_cause,
                "troubleshooting": troubleshooting,
                "full_user_msg": user_msg[:500],
                "full_assistant_msg": assistant_msg[:500],
            }
        )

    embeddings = []
    for text in tqdm(training_texts, desc="Embeddings"):
        embeddings.append(embed_text(text))

    TRAINING_EMBEDDINGS = np.vstack(embeddings)
    print(f"Embeddings ready! ({len(TRAINING_INFO)} cases indexed)")


def find_similar_cases(error_code, device_id, temps, humidities, top_k=3):
    """Find similar historical cases using BioBERT"""
    if not HAS_TRAINING_DATA:
        return []

    avg_temp = sum(temps) / len(temps) if temps else 25.0
    avg_humidity = sum(humidities) / len(humidities) if humidities else 50.0

    query_text = (
        f"Error: {error_code}, Device: {device_id}, "
        f"Temp: {avg_temp:.1f}°C, Humidity: {avg_humidity:.1f}%"
    )

    query_embedding = embed_text(query_text).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, TRAINING_EMBEDDINGS)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    similar_cases = []
    for idx in top_indices:
        info = TRAINING_INFO[idx]
        similar_cases.append(
            {
                "similarity": float(similarities[idx]),
                "error_code": info["error_code"],
                "device_type": info["device_type"],
                "temperature": info["temperature"],
                "humidity": info["humidity"],
                "severity": info["severity"],
                "fda_code": info["fda_code"],
                "root_cause": info["root_cause"],
                "troubleshooting": info["troubleshooting"],
            }
        )

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
            api_key=POE_API_KEY,
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


async def analyze_with_rag_fewshot(all_logs: List[dict]):
    """Main analysis function using RAG + Few-Shot"""

    error_log = all_logs[2]
    device_id = error_log["device_id"]

    temps = [log["sensor_temp_c"] for log in all_logs]
    humidities = [log["sensor_humidity"] for log in all_logs]

    temp_analysis = {
        "min": min(temps),
        "max": max(temps),
        "avg": sum(temps) / len(temps),
        "trend": "rising"
        if temps[-1] > temps[0]
        else "falling"
        if temps[-1] < temps[0]
        else "stable",
    }

    humidity_analysis = {
        "min": min(humidities),
        "max": max(humidities),
        "avg": sum(humidities) / len(humidities),
        "trend": "rising"
        if humidities[-1] > humidities[0]
        else "falling"
        if humidities[-1] < humidities[0]
        else "stable",
    }

    similar_cases = find_similar_cases(
        error_log.get("maude_error_code") or error_log.get("device_event_code"),
        device_id,
        temps,
        humidities,
        top_k=3,
    )

    prompt = """You are a medical device diagnostic AI assistant specialized in FDA-regulated equipment.

╔═══════════════════════════════════════════════════════════════════╗
║                    FEW-SHOT EXAMPLES                               ║
╚═══════════════════════════════════════════════════════════════════╝

IMPORTANT: You receive TWO types of data:
1. Device messages/error codes (from the medical device itself - may include internal temps like "77°C")
2. Room sensor data (temperature & humidity from ESP32 room sensors)

Your job: Cross-reference device messages with room conditions to diagnose root cause.
"""

    # (Prompt body omitted here for brevity; keep your existing examples and analysis text)

    # Add RAG context if available
    if similar_cases:
        prompt += """
╔═══════════════════════════════════════════════════════════════════╗
║              SIMILAR HISTORICAL CASES (RAG)                        ║
╚═══════════════════════════════════════════════════════════════════╝

"""
        for i, case in enumerate(similar_cases, 1):
            prompt += (
                f"Case {i} (Similarity: {case['similarity']:.1%}):\n"
                f"- Error: {case['error_code']} | Device: {case['device_type']}\n"
                f"- Temperature: {case['temperature']:.1f}°C | "
                f"Humidity: {case['humidity']:.1f}%\n"
                f"- Severity: {case['severity']} | FDA Code: {case['fda_code']}\n"
                f"- Root Cause: {case['root_cause']}\n"
                f"- Troubleshooting: {case['troubleshooting']}\n\n"
            )

    prompt += f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    CURRENT CASE TO ANALYZE                         ║
╚═══════════════════════════════════════════════════════════════════╝

**Device ID:** {device_id}
**Error Code:** {error_log.get("maude_error_code") or error_log.get("device_event_code")}
**Error Message:** {error_log["device_message"]}
**Error Time:** {error_log["device_timestamp"]}
**Subsystem:** {error_log.get("device_subsystem", "Unknown")}

**Temperature Analysis:**
- Range: {temp_analysis["min"]:.1f}°C - {temp_analysis["max"]:.1f}°C
- Average: {temp_analysis["avg"]:.1f}°C
- Trend: {temp_analysis["trend"].upper()}
- Samples: {len(temps)}

**Humidity Analysis:**
- Range: {humidity_analysis["min"]:.1f}% - {humidity_analysis["max"]:.1f}%
- Average: {humidity_analysis["avg"]:.1f}%
- Trend: {humidity_analysis["trend"].upper()}
- Samples: {len(humidities)}

**Log Sequence ({len(all_logs)} logs total):**
"""

    for i, log in enumerate(all_logs):
        log_label = "BEFORE" if i < 2 else "ERROR" if i == 2 else "AFTER"
        level = log.get("device_level", "INFO")
        message = log["device_message"]
        temp = log["sensor_temp_c"]
        humidity = log["sensor_humidity"]

        if i == 2:
            prompt += f"\n**>>> [{log_label}] <<<**\n"

        prompt += (
            f"  [{level}] {message} (Temp: {temp:.1f}°C, Humidity: {humidity:.1f}%)\n"
        )

    prompt += """
╔═══════════════════════════════════════════════════════════════════╗
║                    YOUR DIAGNOSIS                                  ║
╚═══════════════════════════════════════════════════════════════════╝

**CRITICAL: Cross-Reference Device Messages with Room Sensors**

Device-Specific Expected Operating Ranges:
- VENTILATORS: Normal = ~45°C internal, Overheating = >70°C
- MONITORS/IMAGING: Normal = 35-50°C, Overheating = >65°C
- Room conditions should be: 18-28°C, 30-70% humidity

[... keep your existing diagnostic instructions here ...]
"""

    print("\n[LLM] Querying GPT-4.x with RAG + Few-Shot...")
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


def save_diagnostic_report(all_logs: List[dict], diagnosis: str, similar_cases: list):
    """Save complete report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_rag_fewshot_{timestamp}.txt"

    error_log = all_logs[2]

    with open(filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MEDICAL DEVICE DIAGNOSTIC REPORT\n")
        f.write("Model: RAG + Few-Shot with GPT + BioBERT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Report Generated: {datetime.now()}\n")
        f.write(f"Device ID: {error_log['device_id']}\n\n")

        f.write("ERROR SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Error Code: {error_log.get('maude_error_code') or error_log.get('device_event_code')}\n"
        )
        f.write(f"Message: {error_log['device_message']}\n")
        f.write(f"Timestamp: {error_log['device_timestamp']}\n")
        f.write(f"Subsystem: {error_log.get('device_subsystem', 'Unknown')}\n\n")

        if similar_cases:
            f.write("SIMILAR HISTORICAL CASES (RAG)\n")
            f.write("-" * 70 + "\n")
            for i, case in enumerate(similar_cases, 1):
                f.write(
                    f"{i}. {case['error_code']} (similarity: {case['similarity']:.1%})\n"
                )
                f.write(
                    f"   Device: {case['device_type']}, Temp: {case['temperature']:.1f}°C, "
                    f"Humidity: {case['humidity']:.1f}%\n"
                )
                f.write(
                    f"   Severity: {case['severity']}, FDA Code: {case['fda_code']}\n"
                )
                f.write(f"   Root Cause: {case['root_cause']}\n\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("LLM DIAGNOSTIC ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(diagnosis)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    return filename


# ==============================================================================
# API ENDPOINTS
# ==============================================================================


@app.post("/api/ingest")
async def ingest_log_raw(request: Request):
    """
    Raw ingest endpoint that:
    - Reads the body as text.
    - Tries json.loads.
    - If it fails at end-of-input, appends '}' once and retries.
    - Then validates against SingleLogPacket and runs existing logic.
    """
    raw = await request.body()
    try:
        raw_text = raw.decode("utf-8")
    except Exception:
        raw_text = raw.decode("utf-8", errors="replace")

    payload_dict = None
    try:
        payload_dict = json.loads(raw_text)
    except JSONDecodeError as e:
        # If error is at the very end, try appending a closing brace
        if e.pos >= len(raw_text) - 1:
            fixed_text = raw_text + "}"
            try:
                payload_dict = json.loads(fixed_text)
                print("[INGEST] JSON parsed after appending '}'")
            except Exception as e2:
                print(f"[INGEST] Still could not parse JSON after fix: {e2}")
                raise
        else:
            print(f"[INGEST] JSON decode error (no fix attempted): {e}")
            raise

    # Now validate/parse with Pydantic
    packet = SingleLogPacket(**payload_dict)

    device_id = packet.log["device_id"]

    if device_id not in active_sessions:
        active_sessions[device_id] = []
        print(f"\n[SESSION] Started new session for device: {device_id}")

    active_sessions[device_id].append(packet.log)

    print(f"[LOG {packet.log_index}/{packet.log_count}] Received from {device_id}")
    print(f"  Level: {packet.log['device_level']}")
    print(f"  Message: {packet.log['device_message']}")
    print(
        f"  Temp: {packet.log['sensor_temp_c']:.1f}°C, "
        f"Humidity: {packet.log['sensor_humidity']:.1f}%"
    )

    # If we've received all logs for this episode
    expected = packet.log_count
    if len(active_sessions[device_id]) >= expected:
        print("\n" + "=" * 70)
        print("ALL LOGS RECEIVED - STARTING ANALYSIS")
        print("=" * 70)

        all_logs = active_sessions[device_id][:expected]
        error_log = all_logs[2]

        temps = [log["sensor_temp_c"] for log in all_logs]
        humidities = [log["sensor_humidity"] for log in all_logs]

        similar_cases = find_similar_cases(
            error_log.get("maude_error_code") or error_log.get("device_event_code"),
            device_id,
            temps,
            humidities,
            top_k=3,
        )

        if similar_cases:
            print(f"\nFound {len(similar_cases)} similar cases:")
            for case in similar_cases:
                print(
                    f"  • {case['error_code']} (similarity: {case['similarity']:.1%})"
                )

        diagnosis = await analyze_with_rag_fewshot(all_logs)

        print("\n" + "=" * 70)
        print("DIAGNOSIS COMPLETE")
        print("=" * 70)

        filename = save_diagnostic_report(all_logs, diagnosis, similar_cases)
        print(f"✓ Report saved: {filename}\n")

        del active_sessions[device_id]

        return {
            "status": "complete",
            "diagnosis": diagnosis,
            "report_file": filename,
            "model": "RAG + Few-Shot (BioBERT + GPT)",
            "similar_cases_used": len(similar_cases),
            "similar_cases": similar_cases,
            "approach": "hybrid_rag_fewshot",
        }
    else:
        return {
            "status": "buffering",
            "received": len(active_sessions[device_id]),
            "waiting_for": expected,
            "device_id": device_id,
        }


@app.get("/")
def root():
    return {
        "service": "Medical Device Diagnostic - RAG + Few-Shot",
        "model": "BioBERT (RAG) + GPT (Few-Shot)",
        "training_samples": len(training_data) if HAS_TRAINING_DATA else 0,
        "active_sessions": len(active_sessions),
        "status": "operational",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "approach": "RAG + Few-Shot",
        "active_sessions": len(active_sessions),
    }


@app.get("/sessions")
def list_sessions():
    return {
        "active_sessions": list(active_sessions.keys()),
        "session_details": {
            device_id: {"logs_received": len(logs)}
            for device_id, logs in active_sessions.items()
        },
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Medical Device Diagnostic Server - RAG + Few-Shot")
    print("Updated Schema: Sequential Log Ingestion (per-log)")
    print("=" * 70)
    print(f"\nLLM Model: {LLM_MODEL}")
    print(f"BioBERT Model: {MODEL_NAME}")
    print(f"Training Data: {TRAINING_DATA_PATH}")
    print(f"Training Samples: {len(training_data) if HAS_TRAINING_DATA else 0}")
    print("\nEndpoints:")
    print("  POST /api/ingest   - Receive individual logs from ESP32")
    print("  GET  /             - Server info")
    print("  GET  /health       - Health check")
    print("  GET  /sessions     - Active sessions debug info")
    print("\nStarting server on http://0.0.0.0:8001")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
