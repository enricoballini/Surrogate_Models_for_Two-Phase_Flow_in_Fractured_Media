#!/usr/bin/env bash
# Run the whole benchmark: data, reference model, metrics and figures.

set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
mkdir -p results

# 1. Data. The HDF5 files are part of the repository: they are extracted from the
#    original simulation files, which are not public, only if they are missing.
if [ ! -f data/train.h5 ]; then
    python3 src/data/extract_data.py
fi

# 2. Model and training. Replace these scripts to test your own surrogate model.
python3 model_and_training/train.py
python3 model_and_training/predict.py
python3 model_and_training/plot_losses.py

# 3. Metrics and figures of the predictions saved in the results folder.
python3 src/metrics/compute_metrics.py
python3 src/plots/plot_reference_fields.py
python3 src/plots/plot_error_boxplot.py
python3 src/plots/plot_error_in_time.py
python3 src/plots/plot_fields_on_mesh.py
