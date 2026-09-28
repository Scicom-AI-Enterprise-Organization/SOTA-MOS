"""Predict MOS for audio files at any sampling rate, averaging one or more trained runs.

  uv run python -m sotamos.predict --runs exp/cv/ours_base/s0_f* --out scores.csv a.wav b.flac ...

Each run directory needs model.pt (train with `save_ckpt: true`). Every file is read at its own
sampling rate. The SSL branch gets the waveform (resampled to 16 kHz, or the native samples for
`input_mode: native`). The mel branch reads the native-rate spectrum on the common 0-24 kHz axis.
The sampling-rate embedding takes the nearest of 16 / 24 / 48 kHz on a log scale, so 22.05 kHz
maps to 24 kHz and 44.1 kHz to 48 kHz. Scores are mixed-rate listening-test MOS.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from sotamos.data import SR_INDEX, TEST_INDEX
from sotamos.features import normalise, raw_features
from sotamos.model import MOSModel


def load_audio(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(np.ascontiguousarray(x.mean(1)))
    r16 = x if sr == 16000 else AF.resample(x, sr, 16000, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                            resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    return x, sr, r16


def nearest_rate(sr):
    return min(SR_INDEX, key=lambda r: abs(math.log(sr / r)))


class Predictor:
    def __init__(self, run_dir, device):
        ckpt = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
        self.cfg, self.stats = ckpt["cfg"], ckpt["spec_stats"]
        self.model = MOSModel(self.cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, x, sr, r16):
        cfg = self.cfg
        wav = x if cfg["input_mode"] == "native" else r16
        b = {"wav": wav[None], "lens": torch.tensor([len(wav)]),
             "sr_idx": torch.tensor([SR_INDEX[nearest_rate(sr)]]), "test_idx": torch.tensor([TEST_INDEX["mix"]])}
        if cfg["mel"] or cfg["mfcc"]:
            mel, mfcc = raw_features(x, sr, cfg["n_mels"], cfg["n_mfcc"], cfg["mel_fmax"], cfg["mel_win_ms"],
                                     cfg["mel_hop_ms"], cfg["mel_fmax_mode"])
            mel, mfcc = normalise(mel, mfcc, self.stats)
            b.update(mel=mel[None], mel_lens=torch.tensor([mel.shape[1]]),
                     mfcc=mfcc[None], mfcc_lens=torch.tensor([mfcc.shape[1]]))
        b = {k: v.to(self.device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            return float(self.model(b)["score"].float().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories with model.pt; scores are averaged")
    ap.add_argument("--out", help="CSV path (default: stdout)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictors = [Predictor(r, device) for r in args.runs]
    lines = ["file,sampling_rate,mos"]
    for f in args.files:
        x, sr, r16 = load_audio(f)
        mos = float(np.mean([p(x, sr, r16) for p in predictors]))
        lines.append(f"{f},{sr},{mos:.4f}")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
