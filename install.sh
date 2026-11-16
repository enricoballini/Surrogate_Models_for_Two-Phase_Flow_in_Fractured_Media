#!/usr/bin/env bash
# Create the Python virtual environment in the .venv folder and install the packages.

set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Done. Activate the environment with: source .venv/bin/activate"
