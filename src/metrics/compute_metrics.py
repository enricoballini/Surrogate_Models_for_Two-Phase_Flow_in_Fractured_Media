"""Compute the scores of the predictions saved in the results folder.

For each split, the errors of pressure and saturation are computed in the time range of
the training set and, if the split has later times, in the extrapolation range. The mass
error and the violation of the saturation bounds are computed on all the times. Each
score is the mean over the scenarios. The results are printed and saved in
results/metrics.json.
"""

import json
import os

import numpy as np

from src.data.dataset import pressure, saturation
from src.metrics.metrics import (
    bound_violation,
    extrapolation_times,
    load_references_and_predictions,
    mass_error,
    relative_error,
)

metrics = {}

for split, evaluation in load_references_and_predictions().items():
    times, fields_reference, fields_predicted, cell_volumes = evaluation
    extrapolation = extrapolation_times(times)
    time_ranges = {"training_times": ~extrapolation}
    if extrapolation.any():
        time_ranges["extrapolation"] = extrapolation

    scores = {}
    for scenario in range(len(fields_reference)):
        reference = fields_reference[scenario].astype(float)
        predicted = fields_predicted[scenario].astype(float)
        volumes = cell_volumes[scenario]

        scenario_scores = {}
        for name, selected in time_ranges.items():
            scenario_scores[f"pressure_error_{name}"] = relative_error(
                pressure(predicted)[selected], pressure(reference)[selected], volumes
            )
            scenario_scores[f"saturation_error_{name}"] = relative_error(
                saturation(predicted)[selected], saturation(reference)[selected], volumes
            )
        scenario_scores["mass_error"] = mass_error(
            saturation(predicted), saturation(reference), volumes
        )
        scenario_scores["saturation_bound_violation"] = bound_violation(
            saturation(predicted), saturation(reference), volumes
        )

        for name, value in scenario_scores.items():
            scores.setdefault(name, []).append(value)

    metrics[split] = {name: float(np.mean(values)) for name, values in scores.items()}

print("Scores, mean over the scenarios")
for split, scores in metrics.items():
    print(split)
    for name, value in scores.items():
        print(f"    {name:36s}{100 * value:8.3f} %")

if os.path.exists("results/efficiency.json"):
    with open("results/efficiency.json") as file:
        metrics["efficiency"] = json.load(file)
    print("efficiency:", metrics["efficiency"])

with open("results/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)
