"""
test_client_logs_server.py

Debug client for the /logs endpoint of logs_server.py.

Assumes:
- logs_server.py is running (e.g., on http://0.0.0.0:8000)
- SQLite DB already populated by logs_pipeline.py
"""

import argparse
import datetime as dt
import json
from typing import Any, Dict

import httpx

from utils import setup_base_logging, get_logger

logger = None


def iso_now_minus(seconds: int) -> str:
    """Return an ISO-8601 UTC timestamp seconds ago, with 'Z' suffix."""
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def print_response(tag: str, resp: httpx.Response) -> None:
    logger.info("=== %s ===", tag)
    logger.info("URL: %s", resp.url)
    logger.info("Status: %d", resp.status_code)
    logger.info("Headers: %s", dict(resp.headers))
    try:
        data: Dict[str, Any] = resp.json()
        logger.info("JSON body:\n%s", json.dumps(data, indent=2))
    except Exception as e:
        logger.warning("Failed to parse JSON body: %s", e)
        logger.info("Raw body:\n%s", resp.text)


def run_queries(base_url: str) -> None:
    client = httpx.Client(timeout=10.0)

    # 1) Basic query: last hour, limit 50
    params_basic = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(3600),
        "limit": 50,
    }
    resp = client.get(f"{base_url}/logs", params=params_basic)
    print_response("BASIC_QUERY", resp)

    # 2) Limit enforcement: last 24h, limit 5
    params_limit = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(24 * 3600),
        "limit": 5,
    }
    resp = client.get(f"{base_url}/logs", params=params_limit)
    print_response("LIMIT_QUERY", resp)

    # 3) Invalid device_id
    params_invalid = {
        "device_id": "unknown-device",
        "since": iso_now_minus(3600),
        "limit": 10,
    }
    resp = client.get(f"{base_url}/logs", params=params_invalid)
    print_response("INVALID_DEVICE_QUERY", resp)

    # 4) Paging example
    params_page1 = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(24 * 3600),
        "limit": 3,
    }
    resp1 = client.get(f"{base_url}/logs", params=params_page1)
    print_response("PAGING_FIRST", resp1)

    try:
        data1 = resp1.json()
    except Exception:
        return

    next_since = data1.get("next_since")
    if not next_since:
        return

    params_page2 = {
        "device_id": "ventilator-01",
        "since": next_since,
        "limit": 10,
    }
    resp2 = client.get(f"{base_url}/logs", params=params_page2)
    print_response("PAGING_SECOND", resp2)

    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://0.0.0.0:8000",
        help="Base URL of the logs server (default: http://0.0.0.0:8000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_base_logging()
    global logger
    logger = get_logger("test_client_logs_server")
    logger.setLevel(args.log_level)

    base_url = args.base_url.rstrip("/")

    run_queries(base_url)


if __name__ == "__main__":
    main()
