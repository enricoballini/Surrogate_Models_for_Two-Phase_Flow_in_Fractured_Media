"""Overview of the data: reference saturation of some training scenarios at some times."""

import matplotlib.pyplot as plt

from src.data.dataset import load_split, saturation
from src.plots.mesh import draw_on_mesh

SCENARIOS = [0, 1, 2]
TIME_INDICES = [0, 20, 40, 67]

parameters, times, fields = load_split("train")
colors = {"cmap": "viridis", "vmin": 0, "vmax": 1}

figure, axes = plt.subplots(
    len(SCENARIOS), len(TIME_INDICES), figsize=(2.5 * len(TIME_INDICES), 2.5 * len(SCENARIOS))
)
for row, scenario in enumerate(SCENARIOS):
    permeability, fault_throw = parameters[scenario]
    axes[row, 0].set_ylabel(f"K45 = {permeability:.2g}\nh = {fault_throw:.3f}")
    for column, time_index in enumerate(TIME_INDICES):
        field = saturation(fields[scenario, time_index])
        image = draw_on_mesh(axes[row, column], field, "train", scenario, **colors)
        if row == 0:
            axes[row, column].set_title(f"t = {times[time_index]:.2f}")

figure.colorbar(image, ax=axes, label="saturation of phase 0")
figure.savefig("results/reference_fields.pdf", bbox_inches="tight")
