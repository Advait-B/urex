#!/bin/bash
# UREX one-time setup for macOS.  Run:  bash ~/Desktop/UREX/urex-main/setup_mac.sh
cd "$(dirname "$0")"
ROOT="$(pwd)"
ok(){ echo "  [OK]   $1"; }
warn(){ echo "  [TODO] $1"; }
echo "== UREX setup =="

# 1. Python 3 + pyserial (in a private venv, so macOS/Homebrew don't complain)
if ! command -v python3 >/dev/null; then
  if command -v brew >/dev/null; then brew install python; else
    warn "Python 3 missing: install from https://www.python.org/downloads/ then re-run"; exit 1; fi
fi
python3 -m venv "$ROOT/.venv" && "$ROOT/.venv/bin/pip" install -q --upgrade pip pyserial numpy scipy scikit-learn opencv-python matplotlib \
  && ok "Python $("$ROOT/.venv/bin/python" -V 2>&1 | cut -d' ' -f2) + numpy/opencv (pipeline) + pyserial $("$ROOT/.venv/bin/python" -c 'import serial;print(serial.__version__)')" \
  || warn "pip install failed (check internet)"

# 2. STM32CubeIDE (needs 2.1.1 or newer for CubeMX2 CMake projects)
IDE=$(ls -d /Applications/STM32CubeIDE*.app 2>/dev/null | head -1)
if [ -n "$IDE" ]; then
  V=$(defaults read "$IDE/Contents/Info" CFBundleShortVersionString 2>/dev/null)
  echo "  found $IDE (version ${V:-unknown})"
  case "$V" in 1.*|2.0.*|2.1.0*) warn "Update CubeIDE to 2.1.1+  (Help > Check for Updates, or st.com)";; *) ok "STM32CubeIDE";; esac
else warn "STM32CubeIDE not found: download 2.1.1+ from st.com (free ST account needed)"; fi

# 3. OpenMV IDE (to copy main.py/link.py onto the camera)
if ls -d /Applications/OpenMV*.app >/dev/null 2>&1; then ok "OpenMV IDE"
elif command -v brew >/dev/null && brew install --cask openmv-ide 2>/dev/null; then ok "OpenMV IDE (installed)"
else warn "OpenMV IDE not found: download from https://openmv.io/pages/download"; fi

# 4. STM32CubeMX2 is only needed if you change pins/peripherals (project is already generated)
ls -d /Applications/STM32CubeMX2*.app >/dev/null 2>&1 && ok "STM32CubeMX2" || echo "  [info] STM32CubeMX2 not found - only needed to change pin setup"

# 5. tidy up the zip Claude left behind
rm -f "$ROOT/../urex-main.zip"

echo; echo "Nucleo serial ports right now:"; ls /dev/tty.usbmodem* 2>/dev/null || echo "  (none - plug the Nucleo CN1 USB in)"
echo; echo "Done. To take photos later:  bash $ROOT/take_photo.sh"
