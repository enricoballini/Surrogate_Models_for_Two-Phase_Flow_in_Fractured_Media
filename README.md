# Surrogate Models for Two-Phase Flow in Fractured Media

Benchmark for surrogate models that predict the pressure and the saturation of two
fluids flowing in a fractured and faulted porous medium.

Two-phase flow through fractured porous media is central to applications such as the
geological storage of CO2, but its simulation is expensive when fractures and faults are
resolved explicitly. Here a heavy fluid lies on top of a light one in a closed domain:
gravity makes them swap their positions, flowing in opposite directions through the rock
and the fractures. The task is to learn a fast surrogate of the map

    (2 parameters, time)  ->  pressure and saturation on the 2727 cells of the domain

from a small number of simulated scenarios.

## Data

The domain is a square with two horizontal rock layers, nine fractures and a fault. The
fractures are lines and their intersections are points, each with its own cells, so the
grid has 2620 triangles in the rock matrix, 101 segments in the fractures and 6 points at
the intersections. All the quantities are non-dimensional. The test case and the
simulations are described in the references below.

| Input  | Description                                           | Range          |
|--------|-------------------------------------------------------|----------------|
| `K45`  | permeability of fractures 4 and 5                     | 5e-3 to 2e3    |
| `h`    | throw of the fault, which shifts its two sides        | -0.05 to 0.05  |
| time   | time since the first saved state                      | 0 to 1.34      |

`K45` is sampled uniformly in its logarithm and `h` uniformly. For comparison, the
permeability is 200 and 20 in the two rock layers and 100 in the other fractures: with
`K45` fractures 4 and 5 go from barriers to conduits. The throw deforms the grid: the
nodes move, the cells stay the same.

The outputs are the pressure and the saturation of the heavy fluid, phase 0, in every
cell.

The data are in HDF5 files in the `data` folder:

| File            | Scenarios | Times | Notes       |
|-----------------|-----------|-------|-------------|
| `train.h5`      | 100       | 68    |             |
| `validation.h5` | 20        | 68    |             |
| `test.h5`       | 20        | 100   | not public  |

Each data file contains:

- `parameters`: shape (scenarios, 2)
- `times`: shape (times,)
- `fields`: shape (scenarios, times, 5454), the pressure on the 2727 cells followed by
  the saturation on the same cells
- `initial_fields`: shape (scenarios, 5454), the fields at the first time
- `cell_volumes`: shape (scenarios, 2727)
- `matrix_nodes`, `matrix_triangles`, `fracture_segments`, `intersection_points`: the
  grid of each scenario
- `scenario_ids`: identifiers of the scenarios in the original simulations

The cells are always in the same order: matrix, fractures, intersections.

The states are saved every 0.02. The time is measured from the first saved state: the
simulation starts 0.02 earlier, with the heavy fluid in the top layer and the light one
in the bottom layer, and this initial condition is not in the data. The fields at the
first time are an input: they are given for the test scenarios too, and a model can
start from them, as the reference model does.

The test scenarios last longer than the others, until time 1.98. Their last 32 times
measure how well a model extrapolates in time.

The test set is hidden. Predictions on the test set are evaluated through the Kaggle
challenge at <https://www.kaggle.com/organizations/sciml-geobench/competitions>.

## Run

Install the packages and run all the steps from the main folder of the repository:

    pip install -r requirements.txt
    ./run.sh

`run.sh` trains the reference model, saves its predictions, computes the metrics and
makes the figures. Everything is written in the `results` folder. The training takes
about three minutes on a laptop GPU. The figures are optional: they are not part of the
assessment, they only help to interpret the results.

To run a single script, set `PYTHONPATH` to the main folder first:

    export PYTHONPATH=$PWD
    python3 model_and_training/train.py

## Test your own model

The folder `model_and_training` contains the reference model, a Dynamical Reduced
Embedding (DRE) written with JAX. Replace it with your model and its training.

Read the data with JAX:

    import jax
    from src.data.jax_dataset import load_jax_samples, minibatches
    inputs, fields = load_jax_samples("train")
    for batch_inputs, batch_fields in minibatches(inputs, fields, 8, jax.random.key(0)):
        ...

or with PyTorch:

    from src.data.torch_dataset import TwoPhaseFlowDataset
    training_data = TwoPhaseFlowDataset("train")
    inputs, fields = training_data[0]

or as NumPy arrays:

    from src.data.dataset import load_samples
    inputs, fields = load_samples("train")

Each sample has 3 inputs, the 2 parameters followed by the time, and the 5454 values of
the fields. A model that advances the fields in time, like the reference model, can read
the data scenario by scenario instead:

    from src.data.dataset import load_split, load_initial_fields
    parameters, times, fields = load_split("train")
    initial_fields = load_initial_fields("train")

Save the predictions of your model for the inputs of a split:

    from src.data.dataset import load_samples, save_predictions
    inputs, _ = load_samples("validation")
    save_predictions("validation", my_model(inputs))

Then compute the metrics, which do not depend on the model:

    python3 src/metrics/compute_metrics.py

Optionally, make the figures. They are not part of the assessment, they only help to
interpret the results:

    python3 src/plots/plot_error_boxplot.py
    python3 src/plots/plot_error_in_time.py
    python3 src/plots/plot_fields_on_mesh.py

## Metrics

Six scores are reported, all relative and all zero for a perfect model. They are defined
in `metrics.md`.

- Pressure error and saturation error: relative L2 error in space and time, with the
  cells weighted by their volume. They are computed in the time range of the training
  set and, separately, in the extrapolation range after it.
- Mass error: relative error of the mass of phase 0 in the domain, which the simulations
  conserve.
- Violation of the saturation bounds: norm of the part of the predicted saturation
  outside [0, 1], relative to the norm of the saturation.

Each score is the mean over the scenarios of a split. Efficiency is measured with the
number of trainable parameters and with the inference time per scenario. Write them in
`results/efficiency.json` as in `model_and_training/predict.py` to have them reported
with the other metrics.

All the metrics are saved in `results/metrics.json`. The reference DRE gives:

| Score [%]                                 | Validation | Test  | First state kept, test |
|-------------------------------------------|------------|-------|------------------------|
| pressure error, training time range       | 0.033      | 0.034 | 0.95                   |
| saturation error, training time range     | 0.91       | 0.88  | 27                     |
| pressure error, extrapolation             |            | 7.9   | 1.7                    |
| saturation error, extrapolation           |            | 22    | 48                     |
| mass error                                | 0.066      | 0.85  | 0                      |
| violation of the saturation bounds        | 0          | 0     | 0                      |

with 579591 trainable parameters and an inference time of about 5 ms per scenario on a
CPU. The last column is a model that predicts no change: it returns the fields at the
first time for all the times. The pressure changes little in time, so a pressure error
is meaningful only when it is well below that column. The errors in the extrapolation
range change with the random seed: over five trainings, from 18% to 26% for the
saturation and from 3% to 13% for the pressure.

## Folders

    data/                 HDF5 files of the benchmark
    model_and_training/   reference model (DRE), its training and its predictions
    src/data/             data reading for NumPy, PyTorch and JAX users
    src/metrics/          metrics of the predictions
    src/plots/            optional figures
    metrics.md            definition of the metrics
    run.sh                runs everything

`src/data/extract_data.py` creates the HDF5 files from the original simulation files,
which are not public.

## References

- E. Ballini, L. Formaggia, A. Fumagalli, E. Keilegavlen, A. Scotti, *A hybrid upwind
  scheme for two-phase flow in fractured porous media*, Computer Methods in Applied
  Mechanics and Engineering 432 (2024). <https://doi.org/10.1016/j.cma.2024.117437>
- E. Ballini, M. Gambarini, A. Fumagalli, L. Formaggia, A. Scotti, P. Zunino, *Model
  reduction of parametric ordinary differential equations via autoencoders:
  representation properties and convergence analysis*, Mathematics in Engineering 8
  (2026). <https://doi.org/10.3934/mine.2026013>
- B. Flemisch et al., *Benchmarks for single-phase flow in fractured porous media*,
  Advances in Water Resources 111 (2018).
  <https://doi.org/10.1016/j.advwatres.2017.10.036>

The first reference describes the simulations, the second the reference model, and the
third the fracture network on which the domain is based.

## License

The code is released under the MIT License, see `LICENSE`.

The data in the `data` folder are released under the Creative Commons Attribution 4.0
International License (CC BY 4.0), see `data/LICENSE`. If you use them, cite the
references above.

Code and data are provided as they are, without warranty of any kind.
