"""Process-pool worker: decode bytes -> (waveform at a trained rate, that rate, 16 kHz waveform).

Runs in separate processes so decoding and resampling never hold the server's GIL. Rates the
model was not trained on go to the nearest trained rate (44.1 -> 48 kHz up, 28 -> 24 kHz down);
the resampler is the one used to build the training features, so served features match offline ones.
"""

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import to_model_rates  # noqa: E402


def init():
    torch.set_num_threads(1)


def decode(raw: bytes):
    """-> (waveform at the nearest trained rate, that rate, 16 kHz copy, input rate)."""
    x, sr_in = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x, sr, r16 = to_model_rates(torch.from_numpy(np.ascontiguousarray(x.mean(1))), sr_in)
    return x.numpy(), int(sr), r16.numpy(), int(sr_in)
