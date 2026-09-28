"""Predict MOS for audio files at any sampling rate.

  uv run python -m sotamos.predict --system results/final/both/system.json a.wav b.flac ...
  uv run python -m sotamos.predict --runs exp/cv/ours_base/s0_f* --out scores.csv a.wav ...

--system runs an exported final ensemble (scripts/export_final.py): fine-tuned members average
their fold checkpoints; probe members average their fold models on frozen SSL features.
--runs averages fine-tuned run directories directly.

Each run directory needs model.pt (train with `save_ckpt: true`). Every file is read at its own
sampling rate. The SSL branch gets the waveform (resampled to 16 kHz, or the native samples for
`input_mode: native`). The mel branch reads the native-rate spectrum on the common 0-24 kHz axis.
Rates the model was not trained on are resampled to the nearest trained rate (44.1 -> 48 kHz up,
28 -> 24 kHz down, 22.05 -> 24 kHz, 8 -> 16 kHz, 96 -> 48 kHz), so the native-rate members always
see a time scale they were trained on. Scores are mixed-rate listening-test MOS.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from sotamos.data import SR_INDEX, TEST_INDEX, nearest_trained_rate, to_model_rates
from sotamos.features import normalise, raw_features
from sotamos.model import MOSModel, SSLBackbone
from sotamos.probes import layer_features, pooled_hidden_states, predict_fold


ROOT = Path(__file__).resolve().parents[1]


def _repo_path(p):
    """system.json stores paths relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_audio(path):
    """(waveform at a trained rate, that rate, 16 kHz copy); see sotamos.data.to_model_rates."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return to_model_rates(torch.from_numpy(np.ascontiguousarray(x.mean(1))), sr)


def nearest_rate(sr):
    return nearest_trained_rate(sr)  # clips are already converted by to_model_rates; this is a guard


class Predictor:
    def __init__(self, run_dir, device):
        ckpt = torch.load(_repo_path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
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


class SystemScorer:
    """Weighted ensemble from results/final/<group>/system.json."""

    def __init__(self, system_path, device):
        self.device, self.members, self.ssl = device, [], {}
        for m in json.load(open(system_path))["members"]:
            if m["type"] == "finetuned":
                self.members.append((m["weight"], m, [Predictor(r, device) for r in m["runs"]]))
            else:
                if m["backbone"] not in self.ssl:
                    self.ssl[m["backbone"]] = SSLBackbone(m["backbone"], layer="weighted").to(device).eval()
                self.members.append((m["weight"], m, torch.load(_repo_path(m["path"]), weights_only=False)["states"]))

    def __call__(self, x, sr, r16):
        feats, total, wsum = {}, 0.0, 0.0
        sr_idx = np.array([SR_INDEX[nearest_rate(sr)]])
        for w, m, obj in self.members:
            if m["type"] == "finetuned":
                score = float(np.mean([p(x, sr, r16) for p in obj]))
            else:
                key = (m["backbone"], m["input_mode"])
                if key not in feats:
                    wav = x if m["input_mode"] == "native" else r16
                    feats[key] = pooled_hidden_states(self.ssl[m["backbone"]], wav, self.device)[None]
                F = layer_features(feats[key], m["layer"], m["window"])
                score = float(np.mean([predict_fold(st, m["kind"], F, sr_idx)[0] for st in obj]))
            total += w * score
            wsum += w
        return total / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", help="exported final ensemble (results/final/<group>/system.json)")
    ap.add_argument("--runs", nargs="+", help="run directories with model.pt; scores are averaged")
    ap.add_argument("--out", help="CSV path (default: stdout)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.system or args.runs, "give --system or --runs"
    predictors = [SystemScorer(args.system, device)] if args.system else [Predictor(r, device) for r in args.runs]
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
