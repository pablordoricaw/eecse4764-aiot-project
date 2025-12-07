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

MAUDE Integration:
    For ERROR_EVENT logs, queries the FDA OpenFDA MAUDE database to enrich
    log entries with real adverse event data. All MAUDE fields are prefixed
    with "maude_" to distinguish from device fields (prefixed "device_").
"""

import argparse
import httpx
from functools import lru_cache
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from logs_db import DEFAULT_DB_PATH, query_logs_since
from utils import setup_base_logging, get_logger


# OpenFDA MAUDE API configuration
OPENFDA_BASE_URL = "https://api.fda.gov/device/event.json"
OPENFDA_TIMEOUT = 10.0  # seconds


# Mapping from ventilator event codes to MAUDE search terms
EVENT_CODE_TO_MAUDE_SEARCH = {
    "TEMP_HIGH_ERROR": "overheating",
    "TEMP_HIGH_WARN": "overheating",
    "SENSOR_READ_FAIL": "sensor+failure",
    "TEMP_FAIL": "temperature+sensor",
    "PRESSURE_FAIL": "pressure+sensor",
    "FLOW_FAIL": "flow+sensor",
    "VOLUME_FAIL": "volume+measurement",
    "O2_FAIL": "oxygen+sensor",
}


@lru_cache(maxsize=128)
def query_maude_api(search_term: str, device_type: str = "ventilator") -> Optional[Dict[str, Any]]:
    """
    Query the OpenFDA MAUDE API for adverse events matching the search term.

    Results are cached to avoid repeated API calls for the same error type.

    Args:
        search_term: The problem/error to search for (e.g., "overheating")
        device_type: The type of medical device (default: "ventilator")

    Returns:
        Dict with MAUDE data or None if query fails
    """
    try:
        # Build search query: device type + problem
        search_query = f"device.generic_name:{device_type}+AND+product_problems:{search_term}"
        url = f"{OPENFDA_BASE_URL}?search={search_query}&limit=5"

        with httpx.Client(timeout=OPENFDA_TIMEOUT) as client:
            response = client.get(url)

            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 404:
                # No results found - not an error, just no matching events
                return None
            else:
                return None

    except httpx.TimeoutException:
        return None
    except Exception:
        return None


def extract_maude_fields(maude_response: Optional[Dict[str, Any]], event_code: str) -> Dict[str, Any]:
    """
    Extract relevant fields from MAUDE API response and format with maude_ prefix.

    Args:
        maude_response: Raw response from OpenFDA API
        event_code: The original event code for fallback error code generation

    Returns:
        Dict with maude_* prefixed fields
    """
    # Default/fallback values
    maude_fields = {
        "maude_error_code": None,
        "maude_product_problems": None,
        "maude_event_type": None,
        "maude_similar_events_count": 0,
        "maude_manufacturer_narrative": None,
        "maude_remedial_action": None,
        "maude_device_class": None,
        "maude_report_date": None,
    }

    # Generate basic error code from event_code
    if event_code:
        if "TEMP_HIGH" in event_code:
            maude_fields["maude_error_code"] = "E-TEMP-HIGH"
        elif "SENSOR" in event_code or "FAIL" in event_code:
            maude_fields["maude_error_code"] = f"E-{event_code.replace('_', '-')}"
        else:
            maude_fields["maude_error_code"] = f"E-{event_code}"

    if not maude_response or "results" not in maude_response:
        return maude_fields

    results = maude_response.get("results", [])
    if not results:
        return maude_fields

    # Get count of similar events from metadata
    meta = maude_response.get("meta", {})
    maude_fields["maude_similar_events_count"] = meta.get("results", {}).get("total", len(results))

    # Extract data from the most recent/first result
    first_result = results[0]

    # Product problems (array of strings)
    product_problems = first_result.get("product_problems", [])
    if product_problems:
        maude_fields["maude_product_problems"] = product_problems

    # Event type (e.g., "Malfunction", "Injury", "Death")
    event_type = first_result.get("event_type")
    if event_type:
        maude_fields["maude_event_type"] = event_type

    # Remedial action taken
    remedial_action = first_result.get("remedial_action", [])
    if remedial_action:
        maude_fields["maude_remedial_action"] = remedial_action

    # Report date
    report_date = first_result.get("date_received") or first_result.get("date_of_event")
    if report_date:
        maude_fields["maude_report_date"] = report_date

    # MDR text narratives (manufacturer description of event)
    mdr_text = first_result.get("mdr_text", [])
    if mdr_text:
        # Find manufacturer narrative if available
        for text_entry in mdr_text:
            text_type = text_entry.get("text_type_code", "")
            text_content = text_entry.get("text", "")
            if text_type in ["Description of Event or Problem", "Manufacturer Narrative"] and text_content:
                # Truncate long narratives
                maude_fields["maude_manufacturer_narrative"] = text_content[:500]
                break

    # Device class from openfda extension
    devices = first_result.get("device", [])
    if devices:
        first_device = devices[0]
        openfda = first_device.get("openfda", {})
        device_class = openfda.get("device_class")
        if device_class:
            maude_fields["maude_device_class"] = device_class

    return maude_fields


def enrich_with_maude(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich an error log record with MAUDE database information.

    Only queries MAUDE for ERROR_EVENT type logs. Other logs get null maude_* fields.

    Args:
        record: Original log record from database

    Returns:
        Dict with maude_* fields to merge into log entry
    """
    event_type = record.get("event_type", "")
    event_code = str(record.get("event_code", "") or "")

    # Only enrich ERROR_EVENT logs
    if event_type != "ERROR_EVENT":
        return {
            "maude_error_code": None,
            "maude_product_problems": None,
            "maude_event_type": None,
            "maude_similar_events_count": None,
            "maude_manufacturer_narrative": None,
            "maude_remedial_action": None,
            "maude_device_class": None,
            "maude_report_date": None,
        }

    # Find appropriate MAUDE search term
    search_term = EVENT_CODE_TO_MAUDE_SEARCH.get(event_code, "malfunction")

    # Query MAUDE API (cached)
    maude_response = query_maude_api(search_term, device_type="ventilator")

    # Extract and return maude_* fields
    return extract_maude_fields(maude_response, event_code)

app = FastAPI()
logger = get_logger("logs_server")


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

    Response schema for each log entry:
    - device_* fields: Data from the medical device logs
    - maude_* fields: Enriched data from FDA MAUDE database (for ERROR_EVENT only)

    The MCU will add sensor_* fields before forwarding to the LLM server.
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
        # Build device_* fields from the log record
        log_entry: Dict[str, Any] = {
            "device_timestamp": r.get("timestamp"),
            "device_level": r.get("level"),
            "device_id": r.get("device_id"),
            "device_event_type": r.get("event_type"),
            "device_event_code": r.get("event_code"),
            "device_message": r.get("message"),
            "device_subsystem": r.get("subsystem"),
        }

        # Add extra device fields if present (sensor readings, etc.)
        for key in ["temp_c", "temp_limit_high_c", "duration_s", "sensor",
                    "airway_pressure_peak", "plateau_pressure", "circuit_flow_l_min",
                    "target_rr", "measured_rr", "target_vt_ml", "measured_vt_ml"]:
            if key in r:
                log_entry[f"device_{key}"] = r.get(key)

        # Enrich with MAUDE data (queries FDA API for ERROR_EVENT logs)
        maude_fields = enrich_with_maude(r)
        log_entry.update(maude_fields)

        logs.append(log_entry)

    # Determine next_since for pagination
    if logs:
        next_since = logs[-1].get("device_timestamp")
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
