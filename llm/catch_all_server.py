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
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="MCU Catch-All Ingest Server")

# Will be set from CLI
LOG_FILE_PATH = "mcu_payloads.txt"


@app.post("/ingest")
async def ingest_any(request: Request) -> Dict[str, Any]:
    """
    Catch-all endpoint for MCU POST requests.

    - Accepts any body (JSON or not).
    - Logs raw body and, if possible, parsed JSON.
    """
    raw = await request.body()
    timestamp = datetime.now().isoformat()
    headers = dict(request.headers)

    print(
        f"Raw length: {len(raw)} bytes, Content-Length: {headers.get('content-length')}"
    )
    print("\n" + "=" * 70)
    print(f"MCU POST RECEIVED @ {timestamp}")
    print("=" * 70)
    headers = dict(request.headers)
    print(f"Headers: {headers}")
    print("Raw body:")
    try:
        raw_text = raw.decode("utf-8")
        print(raw_text)
    except Exception:
        raw_text = repr(raw)
        print(raw_text)
    print("-" * 70)

    # Try to parse JSON, but don't fail if it isn't JSON
    payload = None
    try:
        payload = json.loads(raw)
        print("Parsed JSON:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[WARN] Could not parse JSON: {e}")

    print("=" * 70 + "\n")

    # Append to text log file if configured
    if LOG_FILE_PATH:
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"MCU POST RECEIVED @ {timestamp}\n")
                f.write("=" * 70 + "\n")
                f.write("Headers:\n")
                for k, v in headers.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\nRaw body:\n")
                f.write(raw_text + "\n")
                if payload is not None:
                    f.write("\nParsed JSON:\n")
                    f.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
                f.write("=" * 70 + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write to log file {LOG_FILE_PATH}: {e}")

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
        default="mcu_payloads.txt",
        help=(
            "Path to text file where payloads will be appended "
            '(default: "mcu_payloads.txt" in current directory)'
        ),
    )
    return parser


if __name__ == "__main__":
    parser = setup_argparser()
    args = parser.parse_args()

    LOG_FILE_PATH = args.log_file

    print("\n" + "=" * 70)
    print("MCU Catch-All Ingest Server")
    print("=" * 70)
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 70 + "\n")

    uvicorn.run("catch_all_server:app", host=args.host, port=args.port, reload=False)
