import argparse
import atexit
import datetime
import json
import logging
import logging.config
import logging.handlers
import os
import random
import time

from enum import Enum
from typing import override

from utils import setup_base_logging

###############################################################################
# Logging Utils
###############################################################################

logger = None


class JsonLineFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        always_fields = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
        }
        if record.exc_info:
            always_fields["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            always_fields["stack_info"] = record.stack_info  # already a string

        message = {}
        for key, attr_name in self.fmt_keys.items():
            if attr_name in always_fields:
                message[key] = always_fields.pop(attr_name)
            else:
                message[key] = getattr(record, attr_name, None)

        message.update(always_fields)
        return json.dumps(message, ensure_ascii=False)


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


class VentilatorLoggerAdapter(logging.LoggerAdapter):
    @override
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("device_id", "ventilator-01")
        extra.setdefault("event_type", "UNKNOWN")
        extra.setdefault("event_code", "NONE")
        extra.setdefault("subsystem", "GENERAL")
        return msg, kwargs


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "jsonl": {
            "()": f"{__name__}.JsonLineFormatter",
            "fmt_keys": {
                "level": "levelname",
                "message": "message",
                "timestamp": "timestamp",
                "device_id": "device_id",
                "event_type": "event_type",
                "event_code": "event_code",
                "subsystem": "subsystem",
            },
        },
    },
    "filters": {
        "no_errors": {
            "()": f"{__name__}.NonErrorFilter",
        },
    },
    "handlers": {
        "jsonl_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "jsonl",
            "filename": "logs/ventilator.latest.log.jsonl",
            "maxBytes": 2_000,
            "backupCount": 50,
            "encoding": "utf-8",
        },
        "queue_handler": {
            "class": "logging.handlers.QueueHandler",
            # Python 3.12+: 'handlers' here are the targets for the QueueListener
            "handlers": ["jsonl_file"],
            "respect_handler_level": True,
        },
    },
    "loggers": {
        # Attach the queue_handler to the ventilator logger only;
        # console handlers come from logs_utils.setup_base_logging().
        "ventilator-01": {
            "level": "DEBUG",
            "handlers": ["queue_handler"],
            "propagate": True,
        },
    },
}


def jsonl_file_namer(default_name: str) -> str:
    dirname, _ = os.path.split(default_name)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    new_basename = f"ventilator.{ts}.log.jsonl"
    return os.path.join(dirname, new_basename)


def setup_logging() -> VentilatorLoggerAdapter:
    """
    Initialize logging for the ventilator:

    - Configure base console logging (stdout/stderr) via logs_utils.setup_base_logging.
    - Add JSONL rotating file handler + QueueHandler/QueueListener.
    - Wrap the ventilator logger with VentilatorLoggerAdapter to inject device fields.
    """
    # 1) Configure shared console logging (stdout/stderr)
    setup_base_logging()

    # 2) Ensure JSONL logs directory exists
    json_handler_cfg = logging_config["handlers"]["jsonl_file"]
    dirname, _ = os.path.split(json_handler_cfg["filename"])
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # 3) Apply ventilator-specific logging config (adds queue_handler + jsonl_file)
    logging.config.dictConfig(logging_config)

    # 4) Configure rotation namer on the RotatingFileHandler inside the QueueListener
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None and hasattr(queue_handler, "listener"):
        for h in queue_handler.listener.handlers:
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.namer = jsonl_file_namer
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)

    # 5) Get the ventilator logger and wrap it with the adapter
    base_logger = logging.getLogger("ventilator-01")
    adapter = VentilatorLoggerAdapter(base_logger, {})

    return adapter


###############################################################################
# Ventilator
###############################################################################


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--num-breaths",
        default=None,
        type=int,
        help="Number of breaths to simulate (default: run indefinitely)",
    )
    parser.add_argument(
        "--fault-interval",
        type=int,
        default=20,
        help="Every N breaths, evaluate whether to inject a sensor fault (default: 20).",
    )
    parser.add_argument(
        "--fault-probability",
        type=float,
        default=0.3,
        help="Probability [0,1] of injecting a sensor fault at each fault interval (default: 0.3).",
    )
    parser.add_argument(
        "--overheat-probability",
        type=float,
        default=0.2,
        help=(
            "Probability [0,1] of entering an overheated internal state on a breath "
            "(default: 0.2). While overheated, internal temp stays above threshold."
        ),
    )
    return parser


TEMP_WARN_THRESH_C = 70.0  # internal hardware over-temperature warn limit
TEMP_ERROR_DURATION_S = 30.0  # sustained over-temperature before ERROR

temp_high_since: float | None = None
overheated: bool = False  # whether the ventilator is in an overheated state


class Subsystem(Enum):
    SYSTEM = 0  # internal software/hardware health
    CONFIG = 1
    CONTROL_LOOP = 2
    SENSORS = 3  # pressure, flow, O2, internal temp sensors


class EventCode(Enum):
    VENT_STARTUP = 0  # log_startup
    VENT_SHUTDOWN = 1  # log_shutdown
    MODE_CHANGE = 2  # log_startup (mode set)
    CTRL_LOOP_STABLE = 3  # log_normal_breath
    SENSOR_READ_OK = 4  # log_normal_breath, temp returns to safe range
    SENSOR_READ_FAIL = 5  # log_sensor_failure
    TEMP_HIGH_WARN = 6  # maybe_log_temperature_alarm (warning)
    TEMP_HIGH_ERROR = 7  # maybe_log_temperature_alarm (error)


def log_startup() -> None:
    logger.info(
        "Ventilator startup sequence complete",
        extra={
            "event_type": "STATE_CHANGE",
            "event_code": EventCode.VENT_STARTUP.name,
            "subsystem": Subsystem.SYSTEM.name,
        },
    )
    logger.info(
        "Mode set to VCV with RR=16, VT=450 mL, PEEP=5 cmH2O, FiO2=0.40",
        extra={
            "event_type": "STATE_CHANGE",
            "event_code": EventCode.MODE_CHANGE.name,
            "subsystem": Subsystem.CONFIG.name,
        },
    )


def log_shutdown() -> None:
    logger.info(
        "Ventilator orderly shutdown",
        extra={
            "event_type": "STATE_CHANGE",
            "event_code": EventCode.VENT_SHUTDOWN.name,
            "subsystem": Subsystem.SYSTEM.name,
        },
    )


def log_normal_breath(breath_idx: int, overheat_probability: float) -> None:
    global overheated

    # --- Control loop “stable” measurement ---
    target_rr = 16
    target_vt_ml = 450
    target_peep = 5.0
    target_fio2 = 0.40

    measured_rr = target_rr + random.uniform(-0.3, 0.3)
    measured_vt_ml = target_vt_ml + random.uniform(-15, 15)
    measured_peep = target_peep + random.uniform(-0.3, 0.3)
    measured_fio2 = target_fio2 + random.uniform(-0.01, 0.01)

    logger.info(
        "Control loop stable for breath %d" % breath_idx,
        extra={
            "event_type": "MEASUREMENT",
            "event_code": EventCode.CTRL_LOOP_STABLE.name,
            "subsystem": Subsystem.CONTROL_LOOP.name,
            "target_rr": target_rr,
            "measured_rr": round(measured_rr, 1),
            "target_vt_ml": target_vt_ml,
            "measured_vt_ml": int(measured_vt_ml),
            "target_peep": target_peep,
            "measured_peep": round(measured_peep, 1),
            "target_fio2": target_fio2,
            "measured_fio2": round(measured_fio2, 2),
        },
    )

    # --- Sensor “OK” snapshot (airway/circuit) ---
    airway_pressure_peak = 22.0 + random.uniform(-1.0, 1.0)
    plateau_pressure = 18.0 + random.uniform(-1.0, 1.0)
    circuit_flow_l_min = 40.0 + random.uniform(-3.0, 3.0)

    logger.info(
        "Sensors within expected range for breath %d" % breath_idx,
        extra={
            "event_type": "MEASUREMENT",
            "event_code": EventCode.SENSOR_READ_OK.name,
            "subsystem": Subsystem.SENSORS.name,
            "airway_pressure_peak": round(airway_pressure_peak, 1),
            "plateau_pressure": round(plateau_pressure, 1),
            "circuit_flow_l_min": round(circuit_flow_l_min, 1),
        },
    )

    # --- Internal temperature monitoring and alarms ---
    now_s = time.time()
    base_internal_temp = 45.0  # °C baseline

    # Simple state machine: occasionally enter/leave an overheated state
    if not overheated and random.random() < overheat_probability:
        overheated = True
    elif overheated and random.random() < 0.05:
        # small chance each breath to cool down and leave overheated state
        overheated = False

    if overheated:
        # While overheated, stay above threshold with some jitter
        measured_temp_c = TEMP_WARN_THRESH_C + random.uniform(0.0, 10.0)
    else:
        # Normal range, below threshold with noise
        measured_temp_c = base_internal_temp + random.uniform(-2.0, 5.0)

    maybe_log_temperature_alarm(breath_idx, measured_temp_c, now_s)


def log_sensor_failure(breath_idx: int, sensor_name: str) -> None:
    logger.warning(
        "Sensor failure on %s at breath %d" % (sensor_name, breath_idx),
        extra={
            "event_type": "ALARM",
            "event_code": EventCode.SENSOR_READ_FAIL.name,
            "subsystem": Subsystem.SENSORS.name,
            "sensor": sensor_name,
        },
    )


def maybe_log_temperature_alarm(
    breath_idx: int, measured_temp_c: float, now_s: float
) -> None:
    global temp_high_since

    if measured_temp_c >= TEMP_WARN_THRESH_C:
        if temp_high_since is None:
            temp_high_since = now_s
            logger.warning(
                "Internal ventilator temperature high on breath %d" % breath_idx,
                extra={
                    "event_type": "ALARM",
                    "event_code": EventCode.TEMP_HIGH_WARN.name,
                    "subsystem": Subsystem.SENSORS.name,
                    "sensor": "TEMP_INTERNAL",
                    "temp_c": round(measured_temp_c, 1),
                    "temp_limit_high_c": TEMP_WARN_THRESH_C,
                    "duration_s": 0.0,
                },
            )
        else:
            duration = now_s - temp_high_since
            logger.warning(
                "Internal ventilator temperature remains high on breath %d"
                % breath_idx,
                extra={
                    "event_type": "ALARM",
                    "event_code": EventCode.TEMP_HIGH_WARN.name,
                    "subsystem": Subsystem.SENSORS.name,
                    "sensor": "TEMP_INTERNAL",
                    "temp_c": round(measured_temp_c, 1),
                    "temp_limit_high_c": TEMP_WARN_THRESH_C,
                    "duration_s": round(duration, 1),
                },
            )
            if duration >= TEMP_ERROR_DURATION_S:
                logger.error(
                    "Internal ventilator temperature above limit for %.1f s on breath %d"
                    % (duration, breath_idx),
                    extra={
                        "event_type": "ERROR_EVENT",
                        "event_code": EventCode.TEMP_HIGH_ERROR.name,
                        "subsystem": Subsystem.SENSORS.name,
                        "sensor": "TEMP_INTERNAL",
                        "temp_c": round(measured_temp_c, 1),
                        "temp_limit_high_c": TEMP_WARN_THRESH_C,
                        "duration_s": round(duration, 1),
                    },
                )
                temp_high_since = None
    else:
        if temp_high_since is not None:
            logger.info(
                "Internal ventilator temperature returned to safe range on breath %d"
                % breath_idx,
                extra={
                    "event_type": "STATE_CHANGE",
                    "event_code": EventCode.SENSOR_READ_OK.name,
                    "subsystem": Subsystem.SENSORS.name,
                    "sensor": "TEMP_INTERNAL",
                    "temp_c": round(measured_temp_c, 1),
                    "temp_limit_high_c": TEMP_WARN_THRESH_C,
                },
            )
        temp_high_since = None


def simulate_normal_operation(
    num_breaths: int | None,
    breath_period_s: float,
    fault_interval: int,
    fault_probability: float,
    overheat_probability: float,
) -> None:
    log_startup()
    start = time.time()
    i = 1
    while True:
        if num_breaths is not None and i > num_breaths:
            break

        # Normal functioning logs
        log_normal_breath(i, overheat_probability=overheat_probability)

        # Fault injection check (generic sensor failures)
        if i % fault_interval == 0:
            if random.random() < fault_probability:
                fault_type = random.choice(
                    [
                        "TEMP_FAIL",
                        "PRESSURE_FAIL",
                        "FLOW_FAIL",
                        "VOLUME_FAIL",
                        "O2_FAIL",
                    ]
                )
                sensor_map = {
                    "TEMP_FAIL": "TEMP_INTERNAL",
                    "PRESSURE_FAIL": "PRESSURE_MAIN",
                    "FLOW_FAIL": "FLOW_MAIN",
                    "VOLUME_FAIL": "VOLUME_CALC",
                    "O2_FAIL": "O2_SENSOR",
                }
                sensor_name = sensor_map[fault_type]
                log_sensor_failure(i, sensor_name)

        elapsed = time.time() - start
        target_next = i * breath_period_s
        sleep_for = max(0.0, target_next - elapsed)
        time.sleep(sleep_for)
        i += 1


def main() -> None:
    parser = setup_argparser()
    args = parser.parse_args()

    global logger
    logger = setup_logging()
    logger.setLevel(args.log_level)

    try:
        simulate_normal_operation(
            num_breaths=args.num_breaths,
            breath_period_s=2.75,
            fault_interval=args.fault_interval,
            fault_probability=args.fault_probability,
            overheat_probability=args.overheat_probability,
        )
    except KeyboardInterrupt:
        log_shutdown()


if __name__ == "__main__":
    main()
