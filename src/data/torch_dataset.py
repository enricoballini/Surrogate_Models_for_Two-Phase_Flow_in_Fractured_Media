"""Benchmark data for PyTorch users."""

import torch
from torch.utils.data import Dataset

from src.data.dataset import load_samples


class TwoPhaseFlowDataset(Dataset):
    """Samples of "train" or "validation".

    Each sample is (inputs, fields):
    inputs: the 2 parameters followed by the time
    fields: the pressure on the 2727 cells followed by the saturation on the same cells
    """

    def __init__(self, split):
        inputs, fields = load_samples(split)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.fields = torch.tensor(fields, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.fields[index]
