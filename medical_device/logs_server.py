"""
logs_server.py

HTTP server exposing ventilator logs from SQLite.

Endpoint:
    GET /logs?device_id=<id>&since=<ISO-8601>&limit=<N>

Response:
    {
      "logs": [ { ... }, ... ],
      "next_since": "<timestamp-of-last-record>" | null
    }
"""

import argparse
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from logs_db import DEFAULT_DB_PATH, query_logs_since
from logs_utils import setup_base_logging, get_logger

app = FastAPI()
logger = get_logger("logs_server")


def normalize_error_code(record: Dict[str, Any]) -> Optional[str]:
    """
    Derive a simple error_code for true error events.

    This is a project-level convention: we loosely treat ERROR_EVENT /
    malfunction-like events as 'device problem' codes similar in spirit
    to MAUDE device problem codes, but not actual MDR codes. [web:251][web:258]
    """
    event_type = record.get("event_type", "")
    event_code = str(record.get("event_code", "") or "")
    if event_type != "ERROR_EVENT":
        return None

    # Simple mapping; extend as needed
    # Example: TEMP_HIGH_ERROR -> E-TEMP-HIGH
    if "TEMP_HIGH_ERROR" in event_code:
        return "E-TEMP-HIGH"
    if "SENSOR_OFFLINE" in event_code:
        return "E-SENSOR-OFFLINE"

    # Default: prefix event_code with E-
    return f"E-{event_code}" if event_code else None


@app.get("/logs")
def get_logs(
    device_id: str = Query(..., description="Device ID, e.g. ventilator-01"),
    since: str = Query(
        ..., description="ISO-8601 timestamp, e.g. 2025-12-03T20:37:00.123Z"
    ),
    limit: int = Query(
        100, gt=0, le=1000, description="Max number of log records to return"
    ),
    db_path: str = DEFAULT_DB_PATH,
):
    """
    Fetch logs for a device since a given timestamp.
    """
    logger.info(f"/logs request device_id={device_id}, since={since}, limit={limit}")

    try:
        rows = query_logs_since(
            device_id=device_id, since_timestamp=since, limit=limit, db_path=db_path
        )
    except Exception as e:
        logger.error(f"Failed to query logs from DB: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    logs: List[Dict[str, Any]] = []
    for r in rows:
        # r is the original payload dict stored by logs_db
        log_entry: Dict[str, Any] = {
            "timestamp": r.get("timestamp"),
            "level": r.get("level"),
            "device_id": r.get("device_id"),
            "device_event_type": r.get("event_type"),
            "device_error_code": r.get("error_code"),
            # event_code in DB maps to event_code in the payload
            "device_event_code": r.get("event_code"),
            "device_message": r.get("message"),
        }
        log_entry["maude_error_code"] = normalize_error_code(r)
        logs.append(log_entry)

    if logs:
        next_since = logs[-1]["timestamp"]
    else:
        next_since = None

    return {"logs": logs, "next_since": next_since}


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="IP address to bind the HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP server (default: 8000)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    return parser


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    setup_base_logging()
    global logger
    logger = get_logger("logs_server")
    logger.setLevel(args.log_level)

    logger.info(
        f"Starting logs_server on {args.host}:{args.port} with db_path={args.db_path}"
    )

    # Pass db_path via app state if you want multiple DBs; for now, rely on default arg
    uvicorn.run(
        "logs_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
