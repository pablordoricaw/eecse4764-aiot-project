"""
catch_all_server.py

Simple HTTP server to receive POST payloads from the MCU and log them.

Responsibilities:
- Expose a generic POST /ingest endpoint.
- Accept arbitrary bodies from the MCU (JSON or not).
- Write each received payload to stdout and append it to a text log file.
- Host/port and log file path are configurable via command-line arguments.
"""

import argparse
import json
from datetime import datetime
from json import JSONDecodeError
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="MCU Catch-All Ingest Server")

# Will be set from CLI
LOG_FILE_PATH = "mcu_payloads.txt"


@app.post("/api/ingest")
async def ingest_any(request: Request) -> Dict[str, Any]:
    raw = await request.body()
    timestamp = datetime.now().isoformat()
    headers = dict(request.headers)

    raw_len = len(raw)
    print(
        f"Raw length: {raw_len} bytes, Content-Length: {headers.get('content-length')}"
    )

    try:
        raw_text = raw.decode("utf-8")
    except Exception as e:
        print(f"[WARN] UTF-8 decode failed: {e}")
        raw_text = raw.decode("utf-8", errors="replace")

    print("\n" + "=" * 70)
    print(f"MCU POST RECEIVED @ {timestamp}")
    print("=" * 70)
    print("Headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")
    print("Raw body:")
    print(raw_text)
    print("-" * 70)

    payload = None
    try:
        payload = json.loads(raw_text)
        print("Parsed JSON (normal):")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except JSONDecodeError as e:
        print(f"[WARN] json.loads failed: {e}")
        # Heuristic: if the error is at the very end, try adding a closing brace
        if e.pos >= len(raw_text) - 1:
            fixed_text = raw_text + "}"
            try:
                payload = json.loads(fixed_text)
                print("[INFO] JSON parsed after appending closing brace '}'.")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception as e2:
                print(f"[WARN] Still could not parse JSON after fix: {e2}")
        else:
            print("[WARN] Not attempting auto-fix; error not at end-of-input.")

    # Append to log file (create if it does not exist)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"MCU POST RECEIVED @ {timestamp}\n")
            f.write("=" * 70 + "\n")
            f.write("Headers:\n")
            for k, v in headers.items():
                f.write(f"  {k}: {v}\n")
            f.write("Raw body:\n")
            f.write(raw_text + "\n")
            if payload is not None:
                f.write("-" * 70 + "\n")
                f.write("Parsed JSON:\n")
                f.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            f.write("=" * 70 + "\n\n")
    except Exception as e:
        print(f"[WARN] Failed to write to log file {log_file_path}: {e}")

    print("=" * 70 + "\n")

    return {"status": "ok", "received_at": timestamp, "parsed": payload is not None}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Simple health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Catch-all server for MCU POST payloads"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind the HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for the HTTP server (default: 8001)",
    )
    parser.add_argument(
        "--log-file",
        default=LOG_FILE_PATH,
        help=(
            "Path to text file where payloads will be appended "
            f'(default: "{LOG_FILE_PATH}" in current directory)'
        ),
    )
    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    global logs_file_path
    log_file_path = args.log_file

    print("\n" + "=" * 70)
    print("MCU Catch-All Ingest Server")
    print("=" * 70)
    print(f"Log file: {log_file_path}")
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 70 + "\n")

    uvicorn.run("catch_all_server:app", host=args.host, port=args.port, reload=False)
