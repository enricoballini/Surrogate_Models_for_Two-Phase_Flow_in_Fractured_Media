"""Read the benchmark data and read or write the predictions of a model, as NumPy arrays.

All the functions expect to be called from the main folder of the repository.
"""

import os

import h5py
import numpy as np

DATA_FOLDER = "data"
RESULTS_FOLDER = "results"

# the cells are always in this order: matrix, fractures, fracture intersections
N_MATRIX_CELLS = 2620
N_FRACTURE_CELLS = 101
N_INTERSECTIONS = 6
N_CELLS = N_MATRIX_CELLS + N_FRACTURE_CELLS + N_INTERSECTIONS


def available_splits():
    """The test file is not public: only the splits found in the data folder are returned."""
    splits = []
    for split in ["train", "validation", "test"]:
        if os.path.exists(f"{DATA_FOLDER}/{split}.h5"):
            splits.append(split)
    return splits


def load_split(split):
    """Data of "train", "validation" or "test".

    parameters: (n_scenarios, 2), the permeability K45 and the fault throw h
    times:      (n_times,)
    fields:     (n_scenarios, n_times, 2 * N_CELLS), the pressure on the cells followed by
                the saturation on the cells. None if the file has no fields.
    """
    with h5py.File(f"{DATA_FOLDER}/{split}.h5", "r") as file:
        parameters = file["parameters"][:]
        times = file["times"][:]
        fields = None
        if "fields" in file:
            fields = file["fields"][:]
    return parameters, times, fields


def load_samples(split):
    """Data of a split as input-output pairs, one for each scenario and time.

    inputs: (n_samples, 3), the 2 parameters followed by the time
    fields: (n_samples, 2 * N_CELLS). None if the file has no fields.
    """
    parameters, times, fields = load_split(split)
    inputs = []
    for scenario_parameters in parameters:
        for time in times:
            inputs.append([*scenario_parameters, time])
    inputs = np.array(inputs)
    if fields is not None:
        fields = fields.reshape(len(inputs), -1)
    return inputs, fields


def pressure(fields):
    """Pressure on the cells: the first half of the last axis of the fields."""
    return fields[..., :N_CELLS]


def saturation(fields):
    """Saturation of phase 0 on the cells: the second half of the last axis of the fields."""
    return fields[..., N_CELLS:]


def load_initial_fields(split):
    """Fields at the first time, (n_scenarios, 2 * N_CELLS).

    They are given for every scenario, also when the other fields are not: a model can
    use them as an input, as the state from which it starts.
    """
    with h5py.File(f"{DATA_FOLDER}/{split}.h5", "r") as file:
        return file["initial_fields"][:]


def load_cell_volumes(split):
    """Volume of the cells, (n_scenarios, N_CELLS). The fault throw changes it."""
    with h5py.File(f"{DATA_FOLDER}/{split}.h5", "r") as file:
        return file["cell_volumes"][:]


def load_mesh(split):
    """Mesh of each scenario. The fault throw moves the nodes, the cells stay the same.

    matrix_nodes:        (n_scenarios, n_nodes, 2), coordinates of the nodes of the matrix
    matrix_triangles:    (N_MATRIX_CELLS, 3), indices of the three nodes of each matrix cell
    fracture_segments:   (n_scenarios, N_FRACTURE_CELLS, 2, 2), end points of each fracture cell
    intersection_points: (n_scenarios, N_INTERSECTIONS, 2)
    """
    with h5py.File(f"{DATA_FOLDER}/{split}.h5", "r") as file:
        return (
            file["matrix_nodes"][:],
            file["matrix_triangles"][:],
            file["fracture_segments"][:],
            file["intersection_points"][:],
        )


def save_predictions(split, fields):
    """Save the fields predicted for the inputs of load_samples(split)."""
    parameters, times, _ = load_split(split)
    fields = np.asarray(fields).reshape(len(parameters), len(times), -1)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    with h5py.File(f"{RESULTS_FOLDER}/predictions_{split}.h5", "w") as file:
        file["fields"] = fields


def load_predictions(split):
    """Predicted fields with the same shape as the fields of load_split(split)."""
    with h5py.File(f"{RESULTS_FOLDER}/predictions_{split}.h5", "r") as file:
        return file["fields"][:]
