"""Causal historical-window decomposition and bidirectional LSTM regression."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / 'output/iwoa_dependencies'
if LOCAL_DEPS.exists():
    sys.path.append(str(LOCAL_DEPS))

import numpy as np
import torch
from torch import nn
from decomposition import ceemdan_window, vmd_window

FIELDS = ['temperature_c', 'pressure_hpa', 'humidity_percent',
          'irradiance_w_m2', 'power_mw']


class PowerBiLSTM(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, values):
        _, (hidden, _) = self.lstm(values)
        correction = self.head(torch.cat([hidden[-2], hidden[-1]], dim=-1)).squeeze(-1)
        # Persistence residual gives each ablation the same initial forecast.
        return values[:, -1, 4] + correction
