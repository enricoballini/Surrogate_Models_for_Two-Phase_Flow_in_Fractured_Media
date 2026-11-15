"""Box plot of the relative errors: one box for each split, one value for each scenario."""

import matplotlib.pyplot as plt

from src.data.dataset import pressure, saturation
from src.metrics.metrics import (
    extrapolation_times,
    load_references_and_predictions,
    relative_error,
)

figure, axes = plt.subplots(1, 2, figsize=(10, 4))

for axis, name, select in [(axes[0], "pressure", pressure), (axes[1], "saturation", saturation)]:
    errors = []
    labels = []

    for split, evaluation in load_references_and_predictions().items():
        times, fields_reference, fields_predicted, cell_volumes = evaluation
        extrapolation = extrapolation_times(times)
        time_ranges = {split: ~extrapolation}
        if extrapolation.any():
            time_ranges[split + "\nextrapolation"] = extrapolation

        for label, selected_times in time_ranges.items():
            errors_of_scenarios = []
            for scenario in range(len(fields_reference)):
                reference = select(fields_reference[scenario, selected_times]).astype(float)
                predicted = select(fields_predicted[scenario, selected_times]).astype(float)
                errors_of_scenarios.append(
                    100 * relative_error(predicted, reference, cell_volumes[scenario])
                )
            errors.append(errors_of_scenarios)
            labels.append(label)

    axis.boxplot(errors, showfliers=False)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_yscale("log")
    axis.set_ylabel(f"relative error of the {name} [%]")
    axis.grid(linestyle="--", alpha=0.5)

figure.savefig("results/error_boxplot.pdf", bbox_inches="tight")
