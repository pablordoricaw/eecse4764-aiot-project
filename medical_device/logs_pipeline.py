"""
logs_pipeline.py

Mini pipeline that ingests ventilator JSONL log files into SQLite.

- Periodically scans the logs/ directory for ventilator log files.
- For each file, tracks the last byte offset ingested.
- Reads any new JSONL lines, parses them, and inserts into the logs DB.
"""

import argparse
import hashlib
import json
import os
import time
from typing import Dict, Tuple

from logs_db import init_db, insert_logs_bulk, DEFAULT_DB_PATH
from utils import setup_base_logging, get_logger

DEFAULT_LOGS_PATH = "./logs"
DEFAULT_OFFSETS_PATH = None  # no offsets file by default
DEFAULT_CURRENT_LOG_FILE = "ventilator.latest.log.jsonl"  # file still being written

# Track how far we've read in each file: {filepath: offset}
file_offsets: Dict[str, int] = {}

logger = None


def compute_log_id(record: Dict[str, object]) -> str:
    """
    Compute a deterministic log_id for a record.

    Uses a stable subset of fields so the same logical log line
    from different files maps to the same ID.
    """
    timestamp = str(record.get("timestamp", ""))
    device_id = str(record.get("device_id", ""))
    event_type = str(record.get("event_type", ""))
    event_code = str(record.get("event_code", ""))
    subsystem = str(record.get("subsystem", ""))
    message = str(record.get("message", ""))

    key = "|".join([timestamp, device_id, event_type, event_code, subsystem, message])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def list_log_files(logs_path: str, current_log_file: str) -> Dict[str, float]:
    """
    Return a mapping of log file path -> mtime for ventilator log files.

    Includes rotated files but explicitly skips the current log file
    that is still being written by the RotatingFileHandler.
    """
    results: Dict[str, float] = {}
    if not os.path.isdir(logs_path):
        logger.debug(f"Logs path {logs_path} is not a directory or does not exist")
        return results

    current_abs = os.path.join(logs_path, current_log_file)

    for name in os.listdir(logs_path):
        if not name.startswith("ventilator.") or not name.endswith(".log.jsonl"):
            continue
        path = os.path.join(logs_path, name)
        if os.path.abspath(path) == os.path.abspath(current_abs):
            # Skip the current (live) log file; only ingest rotated files
            logger.debug(f"Skipping current log file {path} for ingestion")
            continue
        try:
            st = os.stat(path)
        except OSError as e:
            logger.warning(f"Failed to stat log file {path}: {e}")
            continue
        results[path] = st.st_mtime

    logger.debug(f"Found {len(results)} rotated log files under {logs_path}")
    return results


def load_new_lines_from_file(path: str) -> Tuple[int, list]:
    """
    Read any new JSONL lines from `path`, starting at the last stored offset.

    Returns:
        new_offset: new byte offset after reading
        records: list of parsed dicts
    """
    records = []
    last_offset = file_offsets.get(path, 0)
    logger.debug(f"Loading new lines from {path} starting at offset {last_offset}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(last_offset)
            line_number = 0
            for raw_line in f:
                line_number += 1
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON line in %s at line %d: %r",
                        path,
                        line_number,
                        line,
                    )
                    continue

                # Attach deterministic log_id used for deduplication in the DB
                record["log_id"] = compute_log_id(record)
                records.append(record)
            new_offset = f.tell()
    except FileNotFoundError:
        logger.debug(f"Log file {path} disappeared before reading")
        return last_offset, records

    logger.debug(f"Read {len(records)} new records from {path}")
    return new_offset, records


def update_offsets_for_deleted_files(current_files: Dict[str, float]) -> None:
    """
    Remove offsets for files that no longer exist.
    """
    to_remove = [path for path in file_offsets if path not in current_files]
    if to_remove:
        logger.debug(f"Removing offsets for deleted files: {to_remove}")
    for path in to_remove:
        file_offsets.pop(path, None)


def load_offsets(offsets_path: str | None) -> None:
    """
    Load file_offsets from a JSON file if provided and exists.
    """
    global file_offsets
    if not offsets_path:
        logger.debug("No offsets_path provided; starting with empty offsets")
        return
    if not os.path.exists(offsets_path):
        logger.debug(f"Offsets file {offsets_path} does not exist; starting fresh")
        return

    try:
        with open(offsets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_offsets = {str(k): int(v) for k, v in data.items()}
        logger.info(f"Loaded offsets from {offsets_path}")
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"Failed to load offsets from {offsets_path} ({e}), starting fresh"
        )


def save_offsets(offsets_path: str | None) -> None:
    """
    Save current file_offsets to a JSON file and/or print them.

    If offsets_path is None, dump raw JSON to stdout.
    """
    if not offsets_path:
        logger.info("Dumping offsets to stdout as raw JSON")
        print(json.dumps(file_offsets, ensure_ascii=False))
        return

    try:
        with open(offsets_path, "w", encoding="utf-8") as f:
            json.dump(file_offsets, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved offsets to {offsets_path}")
    except OSError as e:
        logger.warning(
            f"Failed to save offsets to {offsets_path} ({e}); dumping to stdout"
        )
        print(json.dumps(file_offsets, ensure_ascii=False))


def run_pipeline(
    poll_interval: float,
    logs_path: str,
    current_log_file: str,
    db_path: str,
) -> None:
    """
    Main loop: periodically scan logs/, ingest new lines into SQLite.
    """
    logger.info(f"Initializing logs database at {db_path}")
    init_db(db_path=db_path)
    logger.info(
        f"Starting pipeline with poll_interval={poll_interval}s, "
        f"logs_path={logs_path}, current_log_file={current_log_file}"
    )

    while True:
        files = list_log_files(logs_path, current_log_file=current_log_file)
        update_offsets_for_deleted_files(files)

        total_new_records = 0
        for path in sorted(files.keys()):
            new_offset, records = load_new_lines_from_file(path)
            if records:
                insert_logs_bulk(records, db_path=db_path)
                total_new_records += len(records)
            file_offsets[path] = new_offset

        if total_new_records:
            logger.info(f"Ingested {total_new_records} new records this cycle")
        else:
            logger.debug("No new records this cycle")

        time.sleep(poll_interval)


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--logs-path",
        default=DEFAULT_LOGS_PATH,
        help=f"Path to location where logs are stored (default: {DEFAULT_LOGS_PATH})",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for scanning log files (default: 2.0)",
    )
    parser.add_argument(
        "--offsets-path",
        default=DEFAULT_OFFSETS_PATH,
        help=(
            "Optional path to a JSON file with initial file offsets. "
            "On shutdown, current offsets will be printed and, if this "
            "path is provided, written back to the file."
        ),
    )
    parser.add_argument(
        "--current-log-file",
        default=DEFAULT_CURRENT_LOG_FILE,
        help=(
            "File name of the current log file that the RotatingFileHandler "
            "is still writing to; this file will be ignored by the pipeline "
            "and only rotated files will be ingested "
            f"(default: {DEFAULT_CURRENT_LOG_FILE})"
        ),
    )
    return parser


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    setup_base_logging()
    global logger
    logger = get_logger("logs_pipeline")
    logger.setLevel(args.log_level)

    logger.info(
        f"Starting logs_pipeline with logs_path={args.logs_path}, db_path={args.db_path}, "
        f"poll_interval={args.poll_interval}, offsets_path={args.offsets_path}, "
        f"current_log_file={args.current_log_file}"
    )

    if not os.path.exists(args.logs_path):
        logger.error(f"{args.logs_path} doesn't exist. Terminating...")
        exit(-1)

    load_offsets(args.offsets_path)

    try:
        run_pipeline(
            poll_interval=args.poll_interval,
            logs_path=args.logs_path,
            current_log_file=args.current_log_file,
            db_path=args.db_path,
        )
    except KeyboardInterrupt:
        logger.info("logs_pipeline: shutting down")
    finally:
        save_offsets(args.offsets_path)


if __name__ == "__main__":
    main()
