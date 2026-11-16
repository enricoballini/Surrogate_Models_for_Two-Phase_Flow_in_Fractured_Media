#!/usr/bin/env bash
# Create the Python virtual environment in the .venv folder and install the packages.

set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# With an NVIDIA GPU, install the CUDA version of jax to train on the GPU.
if nvidia-smi > /dev/null 2>&1; then
    .venv/bin/pip install "jax[cuda12]"
fi

echo "Done. Run the benchmark with: ./run.sh"
