"""Plot reference field, predicted field and their difference on the mesh.

One figure is saved for each chosen scenario and for pressure and saturation, with one
column for each chosen time.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import pressure, saturation
from src.metrics.metrics import load_references_and_predictions
from src.plots.mesh import draw_on_mesh

SPLIT = "validation"
SCENARIOS = [0, 1]
TIME_INDICES = [0, 22, 45, 67]

times, fields_reference, fields_predicted, _ = load_references_and_predictions()[SPLIT]

for name, select in [("pressure", pressure), ("saturation", saturation)]:
    for scenario in SCENARIOS:
        reference = select(fields_reference[scenario, TIME_INDICES])
        predicted = select(fields_predicted[scenario, TIME_INDICES])
        difference = predicted - reference

        both = np.array([reference, predicted])
        field_colors = {"cmap": "viridis", "vmin": both.min(), "vmax": both.max()}
        largest_difference = np.abs(difference).max()
        difference_colors = {
            "cmap": "RdBu_r",
            "vmin": -largest_difference,
            "vmax": largest_difference,
        }

        figure, axes = plt.subplots(
            3, len(TIME_INDICES), figsize=(2.5 * len(TIME_INDICES), 7.5)
        )
        for column, time_index in enumerate(TIME_INDICES):
            axes[0, column].set_title(f"t = {times[time_index]:.2f}")
            draw_on_mesh(axes[0, column], reference[column], SPLIT, scenario, **field_colors)
            field_image = draw_on_mesh(
                axes[1, column], predicted[column], SPLIT, scenario, **field_colors
            )
            difference_image = draw_on_mesh(
                axes[2, column], difference[column], SPLIT, scenario, **difference_colors
            )
        axes[0, 0].set_ylabel("reference")
        axes[1, 0].set_ylabel("prediction")
        axes[2, 0].set_ylabel("difference")

        figure.colorbar(field_image, ax=axes[:2], label=name)
        figure.colorbar(difference_image, ax=axes[2], label="difference")
        file_name = f"results/{name}_on_mesh_{SPLIT}_scenario_{scenario}.pdf"
        figure.savefig(file_name, bbox_inches="tight")
        plt.close(figure)
