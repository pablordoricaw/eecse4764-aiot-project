# EECSE-4764 Artificial Intelligence of Things (AIoT) Project

This repository contains project implementation for EECSE-4764 Artificial Intelligence of Things (AIoT).

**Semester:** Fall 2025

**Instructor**: Xiaofan (Fred) Jiang

**Team Number:** 9

**Team Members:**
- Junhyung (Richard) Oh - jo2837 ([@ohrjh10](https://github.com/ohrjh10))
- Ying-Shun (Jason) Liao - yl6026 ([@jasonnliao](https://github.com/jasonnliao))
- Pablo Ordorica-Wiener - po2311 ([@pablordoricaw](https://github.com/pablordoricaw))
- Rahul Murugan - rmm2292 ([@rahulmurugan](https://github.com/rahulmurugan))

## Quick Command Reference

| Task | Command |
|------|---------|
| Install shared dependencies | `uv sync` |
| Find connected board | `ls -la /dev/tty.*` |
| Open REPL session | `mpfshell --open /dev/tty.*** -nc repl` |
| Upload code to board | `mpfshell -nc "open /dev/tty.***; mput main.py"` |

## Hardware

### Adafruit HUZZAH32 ESP32 V2 Feather Board

This course uses the **Adafruit HUZZAH32 ESP32 V2 Feather Board**, a development board featuring the ESP32-PICO-V3-02 module with built-in Wi-Fi and Bluetooth capabilities.

> [!IMPORTANT]
> Do not confuse this board with the original Adafruit HUZZAH32 ESP32 Feather board. We are using **V2** of the board, which has different specifications and pin configurations.

#### Pinout Reference

For detailed pinout diagrams and pin capabilities, refer to the official Adafruit documentation:
- [HUZZAH32 ESP32 V2 Overview](https://learn.adafruit.com/adafruit-esp32-feather-v2/overview)
- [Pinout Diagram](https://learn.adafruit.com/adafruit-esp32-feather-v2/pinouts)

#### Connecting the Board

1. **USB Connection:** Use a USB-C cable to connect the board to your computer
   - Ensure the cable supports data transfer (not just power/charging)
   - The board will appear as a serial device on your system

2. **Driver Installation:**
   - **macOS/Linux:** Drivers are usually installed automatically
   - **Windows:** You may need to install [CP210x USB to UART drivers](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
3. **Verify Connection:**

Look for `/dev/tty.usbserial-*` (macOS/Linux) with

```bash
ls -la /dev/tty.*
```

#### Upload Python Code to Board

The board contains an internal flash that can store `.py` code files we
upload from our computer. When the microcontroller boots up, it first finds, and if it
exists, executes `boot.py`, then `main.py`.

- `boot.py` is usually used for low-level initial set-ups, like Wi-Fi configuration, etc.
- `main.py` is the main entry point.

To save code in our microcontroller, we simply create an `main.py` and upload it
onto the microcontroller using `mpfshell`, then click the reset button to reboot the
microcontroller for it to start running.

```bash
mpfshell -nc "open tty.***; mput main.py"
```

If you don’t like the name `main.py`, you can name your code `new_name.py`, and simply
put the following line in `main.py`, then upload both files.

```python
import new_name
```

This line will upload all `*.py` in your current folder

```bash
mpfshell -nc "open tty.***; mput .*\.py"
```

More on the usage of `mpfshell` on [https://github.com/wendlers/mpfshell#shell-usage](https://github.com/wendlers/mpfshell#shell-usage)
#### Additional Resources

- [Adafruit HUZZAH32 V2 Guide](https://learn.adafruit.com/adafruit-esp32-feather-v2)
- [ESP32 Technical Reference Manual](https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf)
- [MicroPython ESP32 Documentation](https://docs.micropython.org/en/latest/esp32/quickref.html)

## Software

### Dependencies

For the project there is a single shared `pyproject.toml` file at the repository root to manage dependencies across the three components:

1. Microcontroller code in `mcu/`
2. Medical Device code in `medical_device/`
3. LLM code in `llm/`

> [!NOTE]
> This project uses [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management. Commands below are for `uv`. It is possible to use dependency and virtual environemnt management tools, but those are not listed below.
>
> Run these commands from the root project directory (where `pyproject.toml` is located), not from individual component directories.


#### Shared Dependencies

Install shared dependencies across the three components with:

```bash
uv sync
```


#### Component-Specific Dependencies

Install the component specific dependencies using:

```bash
uv sync --group <component> # Replace <component> with mcu, llm, medical_device
```

And remove specific lab dependencies after you're done working on it:

```bash
uv sync --no-group <component> # Replace <component> with mcu, llm, medical_device
```

### Medical Device Component

The medical device component is made of 3 Python modules:

- ventilator_01.py,
- logs_pipeline.py, and
- logs_server.py

The ventilator is the simulated medical device that writes logs files. These log files are consumed by the logs pipeline which inserts each log record into a SQLite database. Lastly, the logs server responds to HTTP GET requests on its `/logs` endpoint by to serve the log records from the database.

```text

+--------------------+        +----------------------+        +--------------------+
|  ventilator_01.py  |        |   logs_pipeline.py   |        |   logs_server.py   |
|--------------------|        |----------------------|        |--------------------|
| - Simulated        |  JSONL | - Watches logs dir   |  SQL   | - FastAPI server   |
|   ventilator       +------->+ - Parses JSONL       +------->+ - Serves /logs     |
| - Logs to stdout   |  files | - Computes log_id    |  DB    |   from SQLite      |
|   / stderr         |        | - Inserts into DB    |        | - Used by MCU      |
+---------+----------+        +----------+-----------+        +---------+----------+
          |                              |                               |
          |                              |                               |
          v                              v                               v
   +--------------+             +----------------+               +-----------------+
   | logs/        |             | logs.db (SQLite)|              |   HTTP client   |
   |  ventilator. |             |   logs table    |              |  (HTTP GET /logs|
   |  latest.log. |             |  (dedup by      |              |   ?device_id...)|
   |  jsonl       |             |   log_id)       |              +-----------------+
   |  ventilator. |             +-----------------+
   |  <ts>.log.   |
   |  jsonl       |
   +--------------+
```

#### How to Run

First, make sure that you have the dependencies installed:

```bash
uv sync --group medical_device
```

And activating the virtual environment:

```bash
source .venv/bin/activate
```

Then, go into the `medical_device` directory with:

```bash
cd medical_device
```

Then each Python module needs to be run separately, so in 3 different terminal sessions run each one (the order listed below is recommended):

> [!TIP]
> Each Python module has command line arguments to configure certain aspects of their run time behavior. You can get the full usage message by adding a ` -h` to the commands below.
>
> Running the Python modules without command line arguments uses the default values.

1. Run the ventilator

    ```bash
    uv run ventilator_01.py
    ```
2. Run the logs pipeline

    ```bash
    uv run logs_pipeline.py
    ```

3. Run the logs server

    ```bash
    uv run logs_server.py
    ```
