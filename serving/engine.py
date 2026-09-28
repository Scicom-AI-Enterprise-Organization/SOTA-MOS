"""Batched scoring engine for an exported SOTA-MOS system (results/final/<group>/system.json).

A batch is a list of clips (native waveform, sampling rate, 16 kHz waveform) of any lengths.
  * Each SSL backbone is loaded once, truncated one block above the deepest layer any member reads,
    and run once per batch for all its members.
  * Backbones with a layer-norm conv front end (every large model) take padded batches with an
    attention mask; pooling covers valid frames only, so padding does not change the features.
    Group-norm front ends (the base models) normalise over time, so padding would change them;
    those clips run one by one.
  * Probe heads (ridge / kernel ridge, 5 fold models each) are the exact numpy code used offline.
  * Fine-tuned members run their fold checkpoints on the batch with tiled padding, as in training.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sotamos.data import SR_INDEX, TEST_INDEX, collate  # noqa: E402
from sotamos.features import normalise, raw_features  # noqa: E402
from sotamos.model import MOSModel, SSLBackbone  # noqa: E402
from sotamos.predict import nearest_rate  # noqa: E402
from sotamos.probes import predict_fold  # noqa: E402


def _repo_path(p):
    """system.json stores paths relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class Engine:
    def __init__(self, system_path, device="cuda", dtype=torch.bfloat16):
        spec = json.load(open(_repo_path(system_path) if not Path(system_path).exists() else system_path))
        self.device = torch.device(device)
        self.dtype = dtype
        self.members = spec["members"]
        self.need = {}  # backbone -> hidden-state indices any probe member reads
        for m in self.members:
            if m["type"] == "probe":
                lo = max(0, m["layer"] - m["window"] // 2)
                self.need.setdefault(m["backbone"], set()).update(range(lo, lo + m["window"]))
        self.ssl = {}
        for hf, layers in self.need.items():
            # keep one block above the deepest layer read: its output is then an intermediate state,
            # identical to the untruncated model (the final encoder LayerNorm is not applied to it)
            ssl = SSLBackbone(hf, layer="weighted", max_layers=max(layers) + 1).to(self.device).eval()
            ssl.masked = ssl.model.config.feat_extract_norm == "layer"
            self.ssl[hf] = ssl
        self.heads, self.ft = [], []
        for m in self.members:
            if m["type"] == "probe":
                states = torch.load(_repo_path(m["path"]), weights_only=False)["states"]
                self.heads.append((m, states))
            else:
                folds = []
                for run in m["runs"]:
                    ck = torch.load(_repo_path(run) / "model.pt", map_location="cpu", weights_only=False)
                    model = MOSModel(ck["cfg"])
                    model.load_state_dict(ck["state_dict"])
                    folds.append((model.eval().to(self.device), ck["cfg"], ck["spec_stats"]))
                self.ft.append((m, folds))

    def _amp(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda")

    @torch.no_grad()
    def _pooled(self, hf, wavs):
        """Mean and std over valid frames of the needed hidden states: {layer: [B, 2D]}."""
        ssl, layers = self.ssl[hf], sorted(self.need[hf])
        if ssl.do_normalize:
            wavs = [(w - w.mean()) / torch.sqrt(w.var() + 1e-7) for w in wavs]
        groups = [list(range(len(wavs)))] if ssl.masked else [[i] for i in range(len(wavs))]
        out = {layer: [None] * len(wavs) for layer in layers}
        for idx in groups:
            lens = torch.tensor([len(wavs[i]) for i in idx])
            x = torch.zeros(len(idx), int(lens.max()))
            for j, i in enumerate(idx):
                x[j, : lens[j]] = wavs[i]
            kw = {}
            if ssl.masked and len(idx) > 1:
                kw["attention_mask"] = (torch.arange(x.shape[1])[None] < lens[:, None]).long().to(self.device)
            with self._amp():
                hs = ssl.model(x.to(self.device), output_hidden_states=True, **kw).hidden_states
            flens = ssl.model._get_feat_extract_output_lengths(lens).tolist()
            for layer in layers:
                h = hs[layer].float()
                for j, i in enumerate(idx):
                    v = h[j, : flens[j]]
                    out[layer][i] = torch.cat([v.mean(0), v.std(0)]).half().float().cpu()
        return {layer: torch.stack(v).numpy().astype(np.float64) for layer, v in out.items()}

    @torch.no_grad()
    def _finetuned(self, folds, clips):
        preds = []
        for model, cfg, stats in folds:
            wav, lens = collate([n if cfg["input_mode"] == "native" else r for n, _, r in clips], True)
            b = {"wav": wav, "lens": lens,
                 "sr_idx": torch.tensor([SR_INDEX[nearest_rate(sr)] for _, sr, _ in clips]),
                 "test_idx": torch.full((len(clips),), TEST_INDEX["mix"])}
            if cfg["mel"] or cfg["mfcc"]:
                feats = [normalise(*raw_features(n, sr, cfg["n_mels"], cfg["n_mfcc"], cfg["mel_fmax"],
                                                 cfg["mel_win_ms"], cfg["mel_hop_ms"], cfg["mel_fmax_mode"]), stats)
                         for n, sr, _ in clips]
                for k, which in [(0, "mel"), (1, "mfcc")]:
                    fl = torch.tensor([f[k].shape[1] for f in feats])
                    t = torch.zeros(len(feats), feats[0][k].shape[0], int(fl.max()))
                    for i, f in enumerate(feats):
                        t[i, :, : f[k].shape[1]] = f[k]
                    b[which], b[f"{which}_lens"] = t, fl
            b = {k: v.to(self.device) for k, v in b.items()}
            with self._amp():
                preds.append(model(b)["score"].float().cpu().numpy())
        return np.mean(preds, 0)

    def score(self, clips):
        """clips: list of (native float32 tensor, sampling rate, 16 kHz float32 tensor). Returns MOS [B]."""
        sr_idx = np.array([SR_INDEX[nearest_rate(sr)] for _, sr, _ in clips])
        pooled = {}
        for m, _ in self.heads:
            key = (m["backbone"], m["input_mode"])
            if key not in pooled:
                wavs = [n if m["input_mode"] == "native" else r for n, _, r in clips]
                pooled[key] = self._pooled(m["backbone"], wavs)
        total, wsum = np.zeros(len(clips)), 0.0
        for m, states in self.heads:
            lo = max(0, m["layer"] - m["window"] // 2)
            F = np.mean([pooled[(m["backbone"], m["input_mode"])][i] for i in range(lo, lo + m["window"])], 0)
            total += m["weight"] * np.mean([predict_fold(st, m["kind"], F, sr_idx) for st in states], 0)
            wsum += m["weight"]
        for m, folds in self.ft:
            total += m["weight"] * self._finetuned(folds, clips)
            wsum += m["weight"]
        return total / wsum
