"""Plot the relative errors in time: median over the scenarios, with the band from P10 to P90."""

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import load_split, pressure, saturation
from src.metrics.metrics import load_references_and_predictions, relative_error_in_time

_, training_times, _ = load_split("train")
figure, axes = plt.subplots(1, 2, figsize=(10, 4))

for axis, name, select in [(axes[0], "pressure", pressure), (axes[1], "saturation", saturation)]:
    for split, evaluation in load_references_and_predictions().items():
        times, fields_reference, fields_predicted, cell_volumes = evaluation
        errors = []
        for scenario in range(len(fields_reference)):
            reference = select(fields_reference[scenario]).astype(float)
            predicted = select(fields_predicted[scenario]).astype(float)
            volumes = cell_volumes[scenario]
            errors.append(100 * relative_error_in_time(predicted, reference, volumes))

        axis.plot(times, np.median(errors, axis=0), label=split)
        axis.fill_between(
            times,
            np.percentile(errors, 10, axis=0),
            np.percentile(errors, 90, axis=0),
            alpha=0.2,
        )

    axis.axvline(training_times.max(), linestyle=":", color="black", label="last training time")
    axis.set_yscale("log")
    axis.set_xlabel("time")
    axis.set_ylabel(f"relative error of the {name} [%]")
    axis.grid(linestyle="--", alpha=0.5)

axes[0].legend()
figure.savefig("results/error_in_time.pdf", bbox_inches="tight")
