"""
logs_server.py

HTTP server exposing ventilator logs from SQLite.

Endpoints:
- GET /logs?device_id=...&since=...&limit=...

Example:
    uvicorn logs_server:app --reload --port 8080
"""

import argparse
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from logs_db import query_logs_since, DEFAULT_DB_PATH
from utils import setup_base_logging, get_logger

app = FastAPI(title="Ventilator Logs API")
logger = get_logger("logs_server")


@app.get("/logs")
def get_logs(
    device_id: str = Query(..., description="Device ID, e.g. 'ventilator-01'"),
    since: str = Query(
        ..., description="ISO 8601 timestamp; return logs with timestamp >= since"
    ),
    limit: Optional[int] = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of log records to return (default: 100, max: 1000)",
    ),
    db_path: str = Query(DEFAULT_DB_PATH, description="Path to SQLite database file"),
):
    """
    Return log records for a given device since a timestamp.
    """
    logger.info(
        f"GET /logs device_id={device_id}, since={since}, limit={limit}, db_path={db_path}"
    )

    try:
        records = query_logs_since(
            device_id=device_id,
            since_timestamp=since,
            limit=limit,
            db_path=db_path,
        )
    except Exception as e:
        logger.error(f"Failed to query logs from DB {db_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return JSONResponse(
        content={"device_id": device_id, "since": since, "logs": records}
    )


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind the HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for logs_server (default: INFO)",
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

    # Pass db_path via environment or settings if you prefer; here we use a query param default.
    uvicorn.run(
        "logs_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
