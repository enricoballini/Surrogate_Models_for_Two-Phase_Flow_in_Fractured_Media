"""Convert the original simulation files into the HDF5 files of the benchmark.

Only needed by the maintainers: the original files in data/raw are not public.
It writes data/train.h5, data/validation.h5 and data/test.h5.
"""

import h5py
import numpy as np

RAW_FOLDER = "data/raw/case_2_two_phase_flow_fractured_faulted_medium/data"
GEOMETRY_FOLDER = "data/raw/geometry"  # written by data/raw/export_geometry.py

PARAMETER_NAMES = ["K45", "h"]
PARAMETER_DESCRIPTIONS = [
    "permeability of fractures 4 and 5 [-]",
    "throw of the fault [-]",
]

TIME_STEP = 0.02  # between two saved states. The simulations use a step of 0.001.

N_MATRIX_CELLS = 2620
N_FRACTURE_CELLS = 101
N_INTERSECTIONS = 6
N_CELLS = N_MATRIX_CELLS + N_FRACTURE_CELLS + N_INTERSECTIONS

# The simulation files have the area of the matrix cells, the length of the fracture
# cells and 1 for the intersections. The volume also needs the aperture of the fractures.
APERTURE = 1e-2
SPECIFIC_VOLUME = np.concatenate(
    [
        np.ones(N_MATRIX_CELLS),
        APERTURE * np.ones(N_FRACTURE_CELLS),
        APERTURE**2 * np.ones(N_INTERSECTIONS),
    ]
)


def read_scenario(scenario_id):
    """Parameters, fields at all the saved times, cell volumes and mesh of one simulation."""
    folder = f"{RAW_FOLDER}/{scenario_id}"

    log_permeability, fault_throw = np.loadtxt(f"{RAW_FOLDER}/mu_param_{scenario_id}")
    parameters = [np.exp(log_permeability), fault_throw]

    # each state is the pressure on the cells followed by the saturation on the cells
    saved_times = np.atleast_1d(np.loadtxt(f"{folder}/PRUNED_TRAINING_TIMES"))
    fields = [np.load(f"{folder}/solution_{time:.10f}.npy") for time in saved_times]

    cell_volumes = np.load(f"{folder}/volumes_subdomains.npy") * SPECIFIC_VOLUME
    mesh = np.load(f"{GEOMETRY_FOLDER}/{scenario_id}.npz")
    return parameters, np.array(fields), cell_volumes, mesh


def write_split(file_name, scenario_ids):
    parameters, fields, cell_volumes = [], [], []
    matrix_nodes, fracture_segments, intersection_points = [], [], []

    for scenario_id in scenario_ids:
        scenario = read_scenario(scenario_id)
        parameters.append(scenario[0])
        fields.append(scenario[1])
        cell_volumes.append(scenario[2])
        mesh = scenario[3]
        matrix_nodes.append(mesh["matrix_nodes"])
        fracture_segments.append(mesh["fracture_segments"])
        intersection_points.append(mesh["intersection_points"])
        if scenario_id == scenario_ids[0]:
            matrix_triangles = mesh["matrix_triangles"]
        # all the scenarios have the same cells: only the nodes move
        assert np.array_equal(mesh["matrix_triangles"], matrix_triangles)

    fields = np.array(fields, dtype=np.float32)
    assert fields.shape[2] == 2 * N_CELLS
    times = TIME_STEP * np.arange(fields.shape[1])

    with h5py.File(file_name, "w") as file:
        file["scenario_ids"] = scenario_ids
        file["parameters"] = np.array(parameters)
        file["parameters"].attrs["names"] = PARAMETER_NAMES
        file["parameters"].attrs["description"] = PARAMETER_DESCRIPTIONS
        file["times"] = times
        file["times"].attrs["units"] = "non-dimensional"
        file.create_dataset("fields", data=fields, compression="gzip", shuffle=True)
        file["fields"].attrs["axes"] = "scenario, time, value"
        file["fields"].attrs["values"] = "pressure on the cells, then saturation on the cells"
        file.create_dataset("initial_fields", data=fields[:, 0], compression="gzip", shuffle=True)
        file["cell_volumes"] = np.array(cell_volumes)
        file["cell_volumes"].attrs["axes"] = "scenario, cell"
        file["cell_volumes"].attrs["cells"] = "matrix, then fractures, then fracture intersections"
        file["matrix_nodes"] = np.array(matrix_nodes, dtype=np.float32)
        file["matrix_triangles"] = matrix_triangles
        file["fracture_segments"] = np.array(fracture_segments, dtype=np.float32)
        file["intersection_points"] = np.array(intersection_points, dtype=np.float32)
    print(file_name, "-", len(scenario_ids), "scenarios,", len(times), "times")


training = np.loadtxt(f"{RAW_FOLDER}/training_dataset_id").astype(int)
validation = np.loadtxt(f"{RAW_FOLDER}/validation_dataset_id").astype(int)
test = np.loadtxt(f"{RAW_FOLDER}/test_dataset_id").astype(int)

# The test simulations are longer than the others: their last times are not in the
# public files and test the extrapolation in time.
write_split("data/train.h5", training)
write_split("data/validation.h5", validation)
write_split("data/test.h5", test)
