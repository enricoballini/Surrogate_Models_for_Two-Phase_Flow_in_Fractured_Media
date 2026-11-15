"""Scores of the benchmark, defined in metrics.md.

Here a field is the pressure or the saturation of one scenario, with one value for each
time and cell: (n_times, n_cells). In the norms the cells are weighted by their volume.
The times are equally spaced, so they all have the same weight.
"""

import os

import numpy as np

from src.data.dataset import available_splits, load_cell_volumes, load_predictions, load_split


def norm(field, cell_volumes):
    """L2 norm of a field in space and time."""
    return np.sqrt(np.sum(field**2 @ cell_volumes))


def relative_error(predicted, reference, cell_volumes):
    """Relative L2 error of a field in space and time."""
    return norm(predicted - reference, cell_volumes) / norm(reference, cell_volumes)


def relative_error_in_time(predicted, reference, cell_volumes):
    """Relative L2 error of a field in space: one value for each time. Used in the plots."""
    error = np.sqrt((predicted - reference) ** 2 @ cell_volumes)
    return error / np.sqrt(reference**2 @ cell_volumes)


def mass_error(saturation_predicted, saturation_reference, cell_volumes):
    """Relative error of the mass of phase 0 in the domain, over the times."""
    mass_predicted = saturation_predicted @ cell_volumes
    mass_reference = saturation_reference @ cell_volumes
    return np.linalg.norm(mass_predicted - mass_reference) / np.linalg.norm(mass_reference)


def bound_violation(saturation_predicted, saturation_reference, cell_volumes):
    """Norm of the part of the predicted saturation outside [0, 1], over the reference norm."""
    outside = np.maximum(np.maximum(-saturation_predicted, saturation_predicted - 1), 0)
    return norm(outside, cell_volumes) / norm(saturation_reference, cell_volumes)


def extrapolation_times(times):
    """True for the times after the last time of the training set."""
    _, training_times, _ = load_split("train")
    return times > training_times.max()


def load_references_and_predictions():
    """Times, reference fields, predicted fields and cell volumes of every split that has both."""
    evaluations = {}
    for split in available_splits():
        _, times, fields_reference = load_split(split)
        has_predictions = os.path.exists(f"results/predictions_{split}.h5")
        if fields_reference is not None and has_predictions:
            evaluations[split] = (
                times,
                fields_reference,
                load_predictions(split),
                load_cell_volumes(split),
            )
    return evaluations
