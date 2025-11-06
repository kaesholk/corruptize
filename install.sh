#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="venv"

# prefer python3, fall back to python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python not found in PATH." >&2
  exit 1
fi

# create venv if it doesn't exist
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Creating virtual environment in ${VENV_DIR}..."
  "$PY" -m venv "$VENV_DIR"
fi

# activate venv for this script
if [ -f "${VENV_DIR}/bin/activate" ]; then
  echo "Activating virtual environment..."
  # shellcheck disable=SC1091
  . "${VENV_DIR}/bin/activate"
else
  echo "ERROR: activation script not found at ${VENV_DIR}/bin/activate." >&2
  exit 1
fi

# check requirements.txt
if [ ! -f requirements.txt ]; then
  echo "ERROR: requirements.txt not found." >&2
  exit 1
fi

echo "Installing packages from requirements.txt..."
# Use python -m pip to ensure the venv's pip is used
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Setup complete."
