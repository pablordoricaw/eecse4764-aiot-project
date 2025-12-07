"""
test_client_logs_server.py

Simple integration tests for the /logs endpoint of logs_server.py.

Assumes:
- logs_server.py is running (e.g., on http://0.0.0.0:8000)
- SQLite DB already populated by logs_pipeline.py
"""

import argparse
import datetime as dt
import json
import sys
from typing import Any, Dict, List

import httpx

from utils import setup_base_logging, get_logger

logger = None


def iso_now_minus(seconds: int) -> str:
    """Return an ISO-8601 UTC timestamp seconds ago, with 'Z' suffix."""
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=seconds)
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def assert_log_shape(log: Dict[str, Any]) -> None:
    """Basic schema checks for a single log entry (device_* + maude_* fields)."""
    required_device_keys = [
        "device_timestamp",
        "device_level",
        "device_id",
        "device_event_type",
        "device_event_code",
        "device_message",
        "device_subsystem",
    ]
    for k in required_device_keys:
        if k not in log:
            raise AssertionError(f"Missing device key {k} in log entry: {log}")

    if not isinstance(log["device_timestamp"], str):
        raise AssertionError("device_timestamp must be a string")
    if not isinstance(log["device_id"], str):
        raise AssertionError("device_id must be a string")

    maude_keys = [
        "maude_error_code",
        "maude_product_problems",
        "maude_event_type",
        "maude_similar_events_count",
        "maude_manufacturer_narrative",
        "maude_remedial_action",
        "maude_device_class",
        "maude_report_date",
    ]
    for k in maude_keys:
        if k not in log:
            raise AssertionError(f"Missing maude key {k} in log entry: {log}")

    if log["maude_error_code"] is not None and not isinstance(
        log["maude_error_code"], str
    ):
        raise AssertionError("maude_error_code must be string or None")
    if log["maude_product_problems"] is not None and not isinstance(
        log["maude_product_problems"], list
    ):
        raise AssertionError("maude_product_problems must be list or None")
    if log["maude_event_type"] is not None and not isinstance(
        log["maude_event_type"], str
    ):
        raise AssertionError("maude_event_type must be string or None")
    if log["maude_similar_events_count"] is not None and not isinstance(
        log["maude_similar_events_count"], int
    ):
        raise AssertionError("maude_similar_events_count must be int or None")
    if log["maude_manufacturer_narrative"] is not None and not isinstance(
        log["maude_manufacturer_narrative"], str
    ):
        raise AssertionError("maude_manufacturer_narrative must be string or None")
    if log["maude_remedial_action"] is not None and not isinstance(
        log["maude_remedial_action"], list
    ):
        raise AssertionError("maude_remedial_action must be list or None")
    if log["maude_device_class"] is not None and not isinstance(
        log["maude_device_class"], (str, int)
    ):
        raise AssertionError("maude_device_class must be string/int or None")
    if log["maude_report_date"] is not None and not isinstance(
        log["maude_report_date"], str
    ):
        raise AssertionError("maude_report_date must be string or None")


def test_basic_query(base_url: str) -> Dict[str, Any]:
    """Test a simple query with default limit."""
    params = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(3600),
        "limit": 50,
    }
    resp = httpx.get(f"{base_url}/logs", params=params, timeout=5.0)
    if resp.status_code != 200:
        raise AssertionError(f"Expected 200, got {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    logs: List[Dict[str, Any]] = data.get("logs", [])
    next_since = data.get("next_since")

    if not isinstance(logs, list):
        raise AssertionError("Response 'logs' must be a list")
    if logs:
        for log in logs:
            assert_log_shape(log)
        if next_since is None:
            raise AssertionError("next_since should not be None when logs are returned")
    else:
        if next_since is not None and not isinstance(next_since, str):
            raise AssertionError("next_since must be string or None")

    return data


def test_limit_enforced(base_url: str) -> Dict[str, Any]:
    """Ensure the limit parameter caps the number of returned logs."""
    params = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(24 * 3600),
        "limit": 5,
    }
    resp = httpx.get(f"{base_url}/logs", params=params, timeout=5.0)
    if resp.status_code != 200:
        raise AssertionError(f"Expected 200, got {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    logs: List[Dict[str, Any]] = data.get("logs", [])
    if len(logs) > 5:
        raise AssertionError(f"Expected at most 5 logs, got {len(logs)}")
    for log in logs:
        assert_log_shape(log)

    return data


def test_invalid_device(base_url: str) -> Dict[str, Any]:
    """Query with a wrong device_id should return 0 logs, not error."""
    params = {
        "device_id": "unknown-device",
        "since": iso_now_minus(3600),
        "limit": 10,
    }
    resp = httpx.get(f"{base_url}/logs", params=params, timeout=5.0)
    if resp.status_code != 200:
        raise AssertionError(f"Expected 200, got {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    logs: List[Dict[str, Any]] = data.get("logs", [])
    if logs:
        raise AssertionError(
            f"Expected 0 logs for unknown device_id, got {len(logs)} entries"
        )

    return data


def test_paging(base_url: str) -> Dict[str, Any]:
    """
    Test that next_since can be used to page:
    - First call gets some logs and a next_since.
    - Second call with since=next_since should return logs at or after that point.
    """
    result: Dict[str, Any] = {}

    first_params = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(24 * 3600),
        "limit": 3,
    }
    resp1 = httpx.get(f"{base_url}/logs", params=first_params, timeout=5.0)
    if resp1.status_code != 200:
        raise AssertionError(
            f"First call expected 200, got {resp1.status_code}: {resp1.text}"
        )

    data1: Dict[str, Any] = resp1.json()
    logs1: List[Dict[str, Any]] = data1.get("logs", [])
    next_since = data1.get("next_since")
    result["first"] = data1

    if len(logs1) < 1 or not next_since:
        return result

    second_params = {
        "device_id": "ventilator-01",
        "since": next_since,
        "limit": 10,
    }
    resp2 = httpx.get(f"{base_url}/logs", params=second_params, timeout=5.0)
    if resp2.status_code != 200:
        raise AssertionError(
            f"Second call expected 200, got {resp2.status_code}: {resp2.text}"
        )

    data2: Dict[str, Any] = resp2.json()
    logs2: List[Dict[str, Any]] = data2.get("logs", [])

    for log in logs2:
        assert_log_shape(log)
        if log["device_timestamp"] < next_since:
            raise AssertionError(
                f"Paging error: log timestamp {log['device_timestamp']} < next_since {next_since}"
            )

    result["second"] = data2
    return result


def test_error_event_maude_enrichment(base_url: str) -> Dict[str, Any]:
    """
    Ensure that ERROR_EVENT logs are present and have MAUDE enrichment fields populated
    when the OpenFDA API returns data.
    """
    params = {
        "device_id": "ventilator-01",
        "since": iso_now_minus(24 * 3600),
        "limit": 200,
    }
    resp = httpx.get(f"{base_url}/logs", params=params, timeout=10.0)
    if resp.status_code != 200:
        raise AssertionError(f"Expected 200, got {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    logs: List[Dict[str, Any]] = data.get("logs", [])

    error_logs = [l for l in logs if l.get("device_event_type") == "ERROR_EVENT"]

    if not error_logs:
        return {"note": "no ERROR_EVENT logs found in test window", "raw": data}

    err = error_logs[0]
    assert_log_shape(err)

    if err.get("maude_error_code") is None:
        raise AssertionError(
            "Expected maude_error_code to be non-null for ERROR_EVENT log"
        )

    return {"first_error_event": err}


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

    try:
        data_basic = test_basic_query(base_url)
        data_limit = test_limit_enforced(base_url)
        data_invalid = test_invalid_device(base_url)
        data_paging = test_paging(base_url)
        data_error_maude = test_error_event_maude_enrichment(base_url)
    except AssertionError as e:
        logger.error(f"[FAIL] {e}")
        sys.exit(1)

    import logging as _logging

    if logger.isEnabledFor(_logging.DEBUG):
        logger.debug("test_basic_query response:\n%s", json.dumps(data_basic, indent=2))
        logger.debug(
            "test_limit_enforced response:\n%s", json.dumps(data_limit, indent=2)
        )
        logger.debug(
            "test_invalid_device response:\n%s", json.dumps(data_invalid, indent=2)
        )
        logger.debug("test_paging response:\n%s", json.dumps(data_paging, indent=2))
        logger.debug(
            "test_error_event_maude_enrichment response:\n%s",
            json.dumps(data_error_maude, indent=2),
        )

    logger.info("[OK] All /logs tests passed")


if __name__ == "__main__":
    main()
