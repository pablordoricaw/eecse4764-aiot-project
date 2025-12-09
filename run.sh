#!/usr/bin/env bash
# This script launches a tmux session with one window per subsystem
# (ventilator, logs pipeline, logs server, LLM server, and MCU). In each
# window it cd's into the project root, activates the shared uv virtual
# environment, and runs the appropriate Python module or MicroPython REPL,
# so the entire AIoT project comes up end-to-end with a single command.
# run.sh
#
# Usage: ./run_aiot_tmux.sh <mcu-device-name>
# Example: ./run_aiot_tmux.sh tty.usbserial-5A6C0422901

set -euo pipefail

SESSION_NAME="aiot-project"
MCU_DEVICE="${1:-}"

if [ -z "$MCU_DEVICE" ]; then
  echo "Usage: $0 <mcu-device-name>"
  echo "Example: $0 tty.usbserial-5A6C0422901"
  echo "- Device name can be found with \`ls /dev/tty.*\`"
  exit 1
fi

# Ensure we run from repo root (adjust if needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv sync --all-groups

# Start new tmux session with first window (ventilator)
tmux new-session -d -s "$SESSION_NAME" -n "ventilator"

# Window 1: ventilator
tmux send-keys -t "$SESSION_NAME:ventilator" "cd \"$SCRIPT_DIR\" && source .venv/bin/activate && cd medical_device && uv run ventilator_01.py" C-m

# Window 2: logs-pipeline
tmux new-window -t "$SESSION_NAME" -n "logs-pipeline"
tmux send-keys -t "$SESSION_NAME:logs-pipeline" "cd \"$SCRIPT_DIR\" && source .venv/bin/activate && cd medical_device && uv run logs_pipeline.py" C-m

# Window 3: logs-server
tmux new-window -t "$SESSION_NAME" -n "logs-server"
tmux send-keys -t "$SESSION_NAME:logs-server" "cd \"$SCRIPT_DIR\" && source .venv/bin/activate && cd medical_device && uv run logs_server.py" C-m

# Window 4: llm-server
tmux new-window -t "$SESSION_NAME" -n "llm-server"
tmux send-keys -t "$SESSION_NAME:llm-server" "cd \"$SCRIPT_DIR\" && source .venv/bin/activate && cd llm_server && uv run llm_server.py" C-m

# Window 5: mcu (MicroPython REPL)
tmux new-window -t "$SESSION_NAME" -n "mcu"
tmux send-keys -t "$SESSION_NAME:mcu" "cd \"$SCRIPT_DIR\" && source .venv/bin/activate && cd mcu && mpfshell -nc \"open ${MCU_DEVICE}; mput ./*.py\" && mpfshell --open ${MCU_DEVICE} -nc repl" C-m

# Attach to the session
tmux attach-session -t "$SESSION_NAME"
