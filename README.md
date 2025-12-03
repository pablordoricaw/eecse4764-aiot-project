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
import lab1_check1
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
