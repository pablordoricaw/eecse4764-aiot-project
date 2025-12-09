"""
llm_server.py

Medical Device Diagnostic Server - RAG + Few-Shot

Responsibilities:
- Expose an HTTP API to receive enriched error packets from the MCU.
- Parse device_* / maude_* / sensor_* fields and raw sensor readings.
- Build a diagnostic prompt and query an LLM via Poe.
- Save a textual diagnostic report to disk.
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

import fastapi_poe as fp
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ==============================================================================
# CONFIGURATION DATA CLASSES
# ==============================================================================


@dataclass
class AppConfig:
    poe_api_key: str
    llm_model: str
    biobert_model_name: str
    training_data_path: str
    reports_path: str


@dataclass
class RagState:
    tokenizer: Optional[AutoTokenizer] = None
    biobert_model: Optional[AutoModel] = None
    has_training_data: bool = False
    training_data: List[dict] = field(default_factory=list)
    training_embeddings: Optional[np.ndarray] = None
    training_info: List[dict] = field(default_factory=list)


@dataclass
class ServerState:
    config: AppConfig
    rag: RagState
    active_sessions: Dict[str, List[dict]] = field(default_factory=dict)


# ==============================================================================
# DATA MODELS - UPDATED FOR NEW SCHEMA
# ==============================================================================


class SingleLogPacket(BaseModel):
    """New schema - receives one log at a time"""

    log: dict  # Contains all log fields including sensor data
    log_index: int  # 1-5 (or 1-based indexing)
    log_count: int  # Total logs in this episode


# ==============================================================================
# RAG / BIOBERT INITIALIZATION
# ==============================================================================


def load_training_data(path: str) -> (List[dict], bool):
    training_data: List[dict] = []
    has_training_data = False
    print(f"Loading training data from {path}...")
    try:
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                training_data.append(data)
        print(f"Loaded {len(training_data)} training samples")
        has_training_data = True
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")
        training_data = []
        has_training_data = False
    return training_data, has_training_data


def init_rag_state(config: AppConfig) -> RagState:
    rag = RagState()
    print("Loading BioBERT for RAG...")
    rag.tokenizer = AutoTokenizer.from_pretrained(config.biobert_model_name)

    if torch.cuda.is_available():
        rag.biobert_model = AutoModel.from_pretrained(config.biobert_model_name).to(
            "cuda"
        )
        print("BioBERT loaded on: NVIDIA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        rag.biobert_model = AutoModel.from_pretrained(config.biobert_model_name).to(
            "mps"
        )
        print("BioBERT loaded on: Apple GPU")
    else:
        rag.biobert_model = AutoModel.from_pretrained(config.biobert_model_name).to(
            "cpu"
        )
        print("BioBERT loaded on: CPU")

    rag.biobert_model.eval()

    rag.training_data, rag.has_training_data = load_training_data(
        config.training_data_path
    )

    if rag.has_training_data:
        create_training_embeddings(rag)

    return rag


def extract_error_info(user_message: str):
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


def extract_diagnosis_summary(assistant_message: str):
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


def embed_text(rag: RagState, text: str) -> np.ndarray:
    """Generate BioBERT embedding"""
    assert rag.tokenizer is not None and rag.biobert_model is not None
    inputs = rag.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(rag.biobert_model.device)
    with torch.no_grad():
        outputs = rag.biobert_model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return vec


def create_training_embeddings(rag: RagState) -> None:
    """Create and store embeddings for training data"""
    print("Creating embeddings for training data...")
    training_texts: List[str] = []
    rag.training_info = []

    for sample in rag.training_data:
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

        rag.training_info.append(
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

    embeddings: List[np.ndarray] = []
    for text in tqdm(training_texts, desc="Embeddings"):
        embeddings.append(embed_text(rag, text))

    rag.training_embeddings = np.vstack(embeddings)
    print(f"Embeddings ready! ({len(rag.training_info)} cases indexed)")


def find_similar_cases(
    rag: RagState,
    error_code: str,
    device_id: str,
    temps: List[float],
    humidities: List[float],
    top_k: int = 3,
) -> List[dict]:
    """Find similar historical cases using BioBERT"""
    if not rag.has_training_data or rag.training_embeddings is None:
        return []

    avg_temp = sum(temps) / len(temps) if temps else 25.0
    avg_humidity = sum(humidities) / len(humidities) if humidities else 50.0

    query_text = (
        f"Error: {error_code}, Device: {device_id}, "
        f"Temp: {avg_temp:.1f}°C, Humidity: {avg_humidity:.1f}%"
    )

    query_embedding = embed_text(rag, query_text).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, rag.training_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    similar_cases: List[dict] = []
    for idx in top_indices:
        info = rag.training_info[idx]
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


async def get_llm_response(config: AppConfig, prompt: str) -> Optional[str]:
    """Query Poe API"""
    full_response: List[str] = []
    try:
        async for partial in fp.get_bot_response(
            messages=[fp.ProtocolMessage(role="user", content=prompt)],
            bot_name=config.llm_model,
            api_key=config.poe_api_key,
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


async def analyze_with_rag_fewshot(state: ServerState, all_logs: List[dict]) -> str:
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
        state.rag,
        error_log.get("maude_error_code") or error_log.get("device_event_code"),
        device_id,
        temps,
        humidities,
        top_k=3,
    )

    prompt = """You are a medical device diagnostic AI assistant specialized in FDA-regulated equipment.

╔═══════════════════════════════════════════════════════════════════╗
║                    FEW-SHOT EXAMPLES                              ║
╚═══════════════════════════════════════════════════════════════════╝

IMPORTANT: You receive TWO types of data:
1. Device messages/error codes (from the medical device itself - may include internal temps like "77°C")
2. Room sensor data (temperature & humidity from ESP32 room sensors)

Your job: Cross-reference device messages with room conditions to diagnose root cause.
"""

    # (Prompt body omitted here for brevity; keep your existing examples and analysis text)

    if similar_cases:
        prompt += """
╔═══════════════════════════════════════════════════════════════════╗
║              SIMILAR HISTORICAL CASES (RAG)                       ║
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
║                    CURRENT CASE TO ANALYZE                        ║
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
║                    YOUR DIAGNOSIS                                 ║
╚═══════════════════════════════════════════════════════════════════╝

**CRITICAL: Cross-Reference Device Messages with Room Sensors**

Device-Specific Expected Operating Ranges:
- VENTILATORS: Normal = ~45°C internal, Overheating = >70°C
- MONITORS/IMAGING: Normal = 35-50°C, Overheating = >65°C
- Room conditions should be: 18-28°C, 30-70% humidity

[... keep your existing diagnostic instructions here ...]
"""

    print("\n[LLM] Querying GPT with RAG + Few-Shot...")
    response = await get_llm_response(state.config, prompt)

    if response:
        response = extract_from_llm_response(response)
        print(f"[LLM] Analysis complete ({len(response)} characters)")
        return response
    else:
        return "LLM analysis failed"


# ==============================================================================
# REPORT GENERATION
# ==============================================================================


def save_diagnostic_report(
    state: ServerState, all_logs: List[dict], diagnosis: str, similar_cases: List[dict]
) -> str:
    """Save complete report into reports_path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(state.config.reports_path, exist_ok=True)
    filename = os.path.join(
        state.config.reports_path, f"report_rag_fewshot_{timestamp}.txt"
    )

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
# FASTAPI APP FACTORY AND ROUTES
# ==============================================================================


async def process_log_sequence(srv, all_logs):
    print("\n" + "=" * 70)
    print("ALL LOGS RECEIVED - STARTING ANALYSIS")
    print("=" * 70)

    error_log = all_logs[2]
    device_id = all_logs[0]["device_id"]

    temps = [log["sensor_temp_c"] for log in all_logs]
    humidities = [log["sensor_humidity"] for log in all_logs]

    similar_cases = find_similar_cases(
        srv.rag,
        error_log.get("maude_error_code") or error_log.get("device_event_code"),
        device_id,
        temps,
        humidities,
        top_k=3,
    )

    if similar_cases:
        print(f"\nFound {len(similar_cases)} similar cases:")
        for case in similar_cases:
            print(f"  • {case['error_code']} (similarity: {case['similarity']:.1%})")

    diagnosis = await analyze_with_rag_fewshot(srv, all_logs)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

    filename = save_diagnostic_report(srv, all_logs, diagnosis, similar_cases)
    print(f"✓ Report saved: {filename}\n")

    return {
        "status": "complete",
        "diagnosis": diagnosis,
        "report_file": filename,
        "model": "RAG + Few-Shot (BioBERT + GPT)",
        "similar_cases_used": len(similar_cases),
        "similar_cases": similar_cases,
        "approach": "hybrid_rag_fewshot",
    }


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="Medical Device Diagnostic - RAG + Few-Shot")
    app.state.server_state = state

    @app.post("/api/ingest")
    async def ingest_log_raw(request: Request):
        """
        Raw ingest endpoint that:
        - Reads the body as text.
        - Tries json.loads.
        - If it fails at end-of-input, appends '}' once and retries.
        - Then validates against SingleLogPacket and runs existing logic.
        """
        srv: ServerState = request.app.state.server_state

        raw = await request.body()
        try:
            raw_text = raw.decode("utf-8")
        except Exception:
            raw_text = raw.decode("utf-8", errors="replace")

        payload_dict: Optional[dict] = None
        try:
            payload_dict = json.loads(raw_text)
        except JSONDecodeError as e:
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

        packet = SingleLogPacket(**payload_dict)
        device_id = packet.log["device_id"]

        if device_id not in srv.active_sessions:
            srv.active_sessions[device_id] = []
            print(f"\n[SESSION] Started new session for device: {device_id}")

        srv.active_sessions[device_id].append(packet.log)

        print(f"[LOG {packet.log_index}/{packet.log_count}] Received from {device_id}")
        print(f"  Level: {packet.log['device_level']}")
        print(f"  Message: {packet.log['device_message']}")
        print(
            f"  Temp: {packet.log['sensor_temp_c']:.1f}°C, "
            f"Humidity: {packet.log['sensor_humidity']:.1f}%"
        )

        expected = packet.log_count
        if len(srv.active_sessions[device_id]) >= expected:
            logs = srv.active_sessions.pop(device_id)
            asyncio.create_task(process_log_sequence(state, logs[:expected]))
            return {
                "status": "queued",
                "received": expected,
                "device_id": device_id,
                "message": "Log sequence complete; analysis running in background",
            }
        else:
            return {
                "status": "buffering",
                "received": len(srv.active_sessions[device_id]),
                "waiting_for": expected,
                "device_id": device_id,
            }

    @app.get("/")
    def root() -> Dict[str, Any]:
        srv: ServerState = app.state.server_state
        return {
            "service": "Medical Device Diagnostic - RAG + Few-Shot",
            "model": "BioBERT (RAG) + GPT (Few-Shot)",
            "training_samples": len(srv.rag.training_data)
            if srv.rag.has_training_data
            else 0,
            "active_sessions": len(srv.active_sessions),
            "status": "operational",
        }

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        srv: ServerState = app.state.server_state
        return {
            "status": "healthy",
            "approach": "RAG + Few-Shot",
            "active_sessions": len(srv.active_sessions),
        }

    @app.get("/sessions")
    def list_sessions() -> Dict[str, Any]:
        srv: ServerState = app.state.server_state
        return {
            "active_sessions": list(srv.active_sessions.keys()),
            "session_details": {
                device_id: {"logs_received": len(logs)}
                for device_id, logs in srv.active_sessions.items()
            },
        }

    return app


# ==============================================================================
# MAIN / CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Medical Device Diagnostic LLM Server (RAG + Few-Shot)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001)",
    )
    parser.add_argument(
        "--reports-path",
        type=str,
        default="./reports/",
        help="Directory to write diagnostic reports (default: ./reports/)",
    )
    parser.add_argument(
        "--training-data-path",
        type=str,
        default="medical_device_training_with_humidity.jsonl",
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--biobert-model-name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Hugging Face model name for BioBERT",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="GPT-4.1-Mini",
        help="Poe LLM model name",
    )
    parser.add_argument(
        "--poe-api-key",
        type=str,
        default="aA_SPfposL5Zgrm3qft9ufaalxrjpkZdvElonE2lG4w",
        help="Poe API key",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = AppConfig(
        poe_api_key=args.poe_api_key,
        llm_model=args.llm_model,
        biobert_model_name=args.biobert_model_name,
        training_data_path=args.training_data_path,
        reports_path=args.reports_path,
    )

    rag_state = init_rag_state(config)
    server_state = ServerState(config=config, rag=rag_state)
    app = create_app(server_state)

    print("\n" + "=" * 70)
    print("Medical Device Diagnostic Server - RAG + Few-Shot")
    print("Updated Schema: Sequential Log Ingestion (per-log)")
    print("=" * 70)
    print(f"\nLLM Model: {config.llm_model}")
    print(f"BioBERT Model: {config.biobert_model_name}")
    print(f"Training Data: {config.training_data_path}")
    print(
        f"Training Samples: {len(rag_state.training_data) if rag_state.has_training_data else 0}"
    )
    print(f"Reports Path: {config.reports_path}")
    print("\nEndpoints:")
    print("  POST /api/ingest   - Receive individual logs from ESP32")
    print("  GET  /             - Server info")
    print("  GET  /health       - Health check")
    print("  GET  /sessions     - Active sessions debug info")
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
