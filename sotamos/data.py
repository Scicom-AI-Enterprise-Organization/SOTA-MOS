"""Clip table, audio cache, training examples and batching for Track 3."""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from sklearn.model_selection import GroupKFold

SR_INDEX = {16000: 0, 24000: 1, 48000: 2}
# listening tests: the three single-rate tests share one "single" protocol id, the mixed test is "mix"
TEST_INDEX = {"single": 0, "mix": 1}


RATES = (16000, 24000, 48000)
_RESAMPLE = dict(lowpass_filter_width=64, rolloff=0.9475937167399596, resampling_method="sinc_interp_kaiser",
                 beta=14.769656459379492)


def resample(x: torch.Tensor, sr: int, target: int) -> torch.Tensor:
    return x if sr == target else AF.resample(x, sr, target, **_RESAMPLE)


def to_model_rates(x: torch.Tensor, sr: int):
    """Waveform at any rate -> (waveform at a trained rate, that rate, 16 kHz copy).

    Rates the model was trained on (16 / 24 / 48 kHz) pass through. Any other rate is resampled up
    to the next trained rate, so no bandwidth is thrown away (44.1 -> 48 kHz, 22.05 -> 24 kHz,
    32 -> 48 kHz, 8 -> 16 kHz); rates above 48 kHz go down to 48 kHz. The 16 kHz copy is made
    from the original signal.
    """
    target = next((r for r in RATES if r >= sr), RATES[-1])
    return resample(x, sr, target), target, resample(x, sr, 16000)


def load_clips(path="data/clips.csv") -> pd.DataFrame:
    return pd.read_csv(path)


class AudioCache:
    """All 800 clips in memory, at native rate and resampled to 16 kHz (cached on disk)."""

    def __init__(self, clips: pd.DataFrame, cache_path="data/cache/audio.pt"):
        if cache_path and os.path.exists(cache_path):
            try:
                d = torch.load(cache_path)
            except Exception:
                d = None
            if d is not None and set(d["native"]) >= set(clips["clip"]):
                self.native, self.r16 = d["native"], d["r16"]
                return
        self.native, self.r16 = {}, {}
        for r in clips.itertuples():
            x, sr = sf.read(r.path, dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            x = torch.from_numpy(np.ascontiguousarray(x))
            self.native[r.clip] = x
            self.r16[r.clip] = resample(x, sr, 16000)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = f"{cache_path}.tmp{os.getpid()}"
            torch.save({"native": self.native, "r16": self.r16}, tmp)
            os.replace(tmp, cache_path)

    def get(self, clip: str, mode: str) -> torch.Tensor:
        return self.native[clip] if mode == "native" else self.r16[clip]


def sentence_folds(clips: pd.DataFrame, k: int, seed: int = 0) -> dict:
    """Assign each pool sentence to one of k folds (GroupKFold over shuffled sentences)."""
    pool = clips[clips.split == "pool"]
    sents = np.array(sorted(pool.sentence.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sents)
    gkf = GroupKFold(n_splits=k)
    fold_of = {}
    for f, (_, idx) in enumerate(gkf.split(sents, groups=sents)):
        for s in sents[idx]:
            fold_of[s] = f
    return fold_of


@dataclass
class Example:
    clip: str
    sr_idx: int
    test_idx: int
    target: float  # clip MOS in that listening test


def make_examples(clips: pd.DataFrame, labels: str) -> list:
    """labels: 'single' (train labels), 'mix' (dev labels), 'both'."""
    ex = []
    for r in clips.itertuples():
        if labels in ("single", "both") and not np.isnan(r.mos_single):
            ex.append(Example(r.clip, SR_INDEX[r.sr], TEST_INDEX["single"], float(r.mos_single)))
        if labels in ("mix", "both") and not np.isnan(r.mos_mix):
            ex.append(Example(r.clip, SR_INDEX[r.sr], TEST_INDEX["mix"], float(r.mos_mix)))
    return ex


def collate(batch_wavs: list, repetitive: bool = True):
    """Pad a list of 1-D waveforms. Repetitive padding tiles each clip (SHEET default), so no
    padding frames need masking; lengths are still returned for masked pooling."""
    lens = torch.tensor([len(w) for w in batch_wavs])
    T = int(lens.max())
    out = torch.zeros(len(batch_wavs), T)
    for i, w in enumerate(batch_wavs):
        if repetitive:
            reps = (T + len(w) - 1) // len(w)
            out[i] = w.repeat(reps)[:T]
        else:
            out[i, : len(w)] = w
    return out, lens


class Batcher:
    """Shuffled mini-batches of examples, loaded from the in-memory cache."""

    def __init__(self, examples, cache: AudioCache, mode: str, batch_size: int, seed: int,
                 repetitive: bool = True):
        self.examples, self.cache, self.mode = examples, cache, mode
        self.bs, self.repetitive = batch_size, repetitive
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            order = self.rng.permutation(len(self.examples))
            for i in range(0, len(order) - self.bs + 1, self.bs):
                yield self.make([self.examples[j] for j in order[i: i + self.bs]])

    def make(self, exs):
        wavs = [self.cache.get(e.clip, self.mode) for e in exs]
        wav, lens = collate(wavs, self.repetitive)
        return {
            "clips": [e.clip for e in exs],
            "wav": wav,
            "lens": lens,
            "sr_idx": torch.tensor([e.sr_idx for e in exs]),
            "test_idx": torch.tensor([e.test_idx for e in exs]),
            "target": torch.tensor([e.target for e in exs], dtype=torch.float32),
        }
