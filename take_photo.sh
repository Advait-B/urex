#!/bin/bash
# Runs get_photo.py with the right Python and finds the Nucleo port automatically.
#   bash take_photo.sh                      save photos / pairs into pc/photos/
#   bash take_photo.sh --analyse            ...and print groove depths after each pair
#   bash take_photo.sh --out photos/calib --shim 0     calibration run (see pipeline/README.md)
#   bash take_photo.sh /dev/tty.usbmodemXXXX --analyse  pick the port yourself
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "$1" && "$1" != -* ]]; then PORT="$1"; shift; else PORT="$(ls /dev/tty.usbmodem* 2>/dev/null | head -1)"; fi
[ -z "$PORT" ] && { echo "No Nucleo found. Plug in CN1 (ST-LINK USB), or pass the port: bash take_photo.sh /dev/tty.usbmodemXXXX"; exit 1; }
cd "$ROOT/urex-camera-photo/urex-camera-photo/pc"   # photos save into pc/photos/
echo "Using $PORT  (Ctrl+C to stop)"
exec "$ROOT/.venv/bin/python" get_photo.py "$PORT" "$@"
