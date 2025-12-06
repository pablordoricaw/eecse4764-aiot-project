"""
logs_db.py

SQLite wrapper for ventilator logs.

Responsibilities:
- Initialize the SQLite database and schema.
- Provide functions to insert log records.
- Provide functions to query logs by device_id and since timestamp.
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DB_PATH = "logs.db"


@contextmanager
def get_conn(db_path: str = DEFAULT_DB_PATH) -> Iterable[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create the logs table and indexes if they don't exist.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_code TEXT NOT NULL,
                subsystem TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        # Deduplication: log_id must be unique
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_logs_log_id
            ON logs (log_id);
            """
        )
        # Index to support queries by device_id + timestamp
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_logs_device_ts
            ON logs (device_id, timestamp);
            """
        )
        conn.commit()


def insert_log(
    record: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Insert a log record into the database.

    Expected fields in `record` (from ventilator JSONL):
    - log_id (deterministic ID computed by the pipeline)
    - timestamp (ISO 8601 string)
    - device_id
    - level
    - event_type
    - event_code
    - subsystem
    - plus any other fields, which will be stored in `payload` as JSON.
    """
    log_id = record.get("log_id")
    timestamp = record.get("timestamp")
    device_id = record.get("device_id")
    level = record.get("level")
    event_type = record.get("event_type")
    event_code = record.get("event_code")
    subsystem = record.get("subsystem")

    if not all(
        [log_id, timestamp, device_id, level, event_type, event_code, subsystem]
    ):
        return

    payload = json.dumps(record, ensure_ascii=False)

    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO logs (
                log_id, timestamp, device_id, level, event_type, event_code, subsystem, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                log_id,
                timestamp,
                device_id,
                level,
                event_type,
                event_code,
                subsystem,
                payload,
            ),
        )
        conn.commit()


def insert_logs_bulk(
    records: Iterable[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Bulk insert multiple log records in a single transaction.
    Records must already contain a deterministic log_id.
    """
    rows: List[Tuple[str, str, str, str, str, str, str, str]] = []
    for record in records:
        log_id = record.get("log_id")
        timestamp = record.get("timestamp")
        device_id = record.get("device_id")
        level = record.get("level")
        event_type = record.get("event_type")
        event_code = record.get("event_code")
        subsystem = record.get("subsystem")

        if not all(
            [log_id, timestamp, device_id, level, event_type, event_code, subsystem]
        ):
            continue

        payload = json.dumps(record, ensure_ascii=False)
        rows.append(
            (
                log_id,
                timestamp,
                device_id,
                level,
                event_type,
                event_code,
                subsystem,
                payload,
            )
        )

    if not rows:
        return

    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO logs (
                log_id, timestamp, device_id, level, event_type, event_code, subsystem, payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()


def query_logs_since(
    device_id: str,
    since_timestamp: str,
    limit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Query logs for a given device_id with timestamp >= since_timestamp,
    ordered by timestamp ascending.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        if limit is not None:
            cur.execute(
                """
                SELECT payload
                FROM logs
                WHERE device_id = ?
                  AND timestamp >= ?
                ORDER BY timestamp ASC, id ASC
                LIMIT ?;
                """,
                (device_id, since_timestamp, limit),
            )
        else:
            cur.execute(
                """
                SELECT payload
                FROM logs
                WHERE device_id = ?
                  AND timestamp >= ?
                ORDER BY timestamp ASC, id ASC;
                """,
                (device_id, since_timestamp),
            )

        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for (payload_str,) in rows:
        try:
            results.append(json.loads(payload_str))
        except json.JSONDecodeError:
            continue

    return results


def query_logs_window(
    device_id: str,
    start_timestamp: str,
    end_timestamp: str,
    limit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Query logs for a given device_id in a [start, end] time window.
    """
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        if limit is not None:
            cur.execute(
                """
                SELECT payload
                FROM logs
                WHERE device_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC, id ASC
                LIMIT ?;
                """,
                (device_id, start_timestamp, end_timestamp, limit),
            )
        else:
            cur.execute(
                """
                SELECT payload
                FROM logs
                WHERE device_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC, id ASC;
                """,
                (device_id, start_timestamp, end_timestamp),
            )

        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for (payload_str,) in rows:
        try:
            results.append(json.loads(payload_str))
        except json.JSONDecodeError:
            continue

    return results


if __name__ == "__main__":
    init_db()
    print(f"Initialized logs database at {DEFAULT_DB_PATH}")
