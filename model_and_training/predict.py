"""Predict the fields with the trained DRE, for every split found in the data folder.

The predictions are saved in the results folder, where the metrics and the plots read
them. The number of trainable parameters and the inference time are saved as well.
"""

import json
import time

import numpy as np

import dre
from src.data.dataset import available_splits, load_initial_fields, load_split, save_predictions

N_TIMING_REPETITIONS = 100

model, scaling = dre.load_model("results/dre.pkl")

for split in available_splits():
    parameters, times, _ = load_split(split)
    initial_fields = load_initial_fields(split)
    fields = []
    for scenario in range(len(parameters)):
        fields.append(
            dre.predict(model, scaling, parameters[scenario], initial_fields[scenario], len(times))
        )
    save_predictions(split, fields)
    print("saved the predictions of the", split, "split")

# inference time: prediction of all the times of one scenario
parameters, times, _ = load_split("train")
initial_fields = load_initial_fields("train")
durations = []
for repetition in range(N_TIMING_REPETITIONS):
    start = time.perf_counter()
    dre.predict(model, scaling, parameters[0], initial_fields[0], len(times))
    durations.append(time.perf_counter() - start)

efficiency = {
    "trainable_parameters": int(dre.n_trainable_parameters(model)),
    "inference_time_per_scenario_seconds": float(np.median(durations)),
}
with open("results/efficiency.json", "w") as file:
    json.dump(efficiency, file, indent=4)
