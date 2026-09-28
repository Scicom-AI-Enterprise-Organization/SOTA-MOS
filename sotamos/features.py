"""Precomputed log-mel and MFCC features on a common Hz axis for every sampling rate.

Every clip gets the same frame rate (hop_ms) and the same mel filterbank over 0..fmax Hz
(default 24 kHz), whatever its sampling rate. Bands above a clip's Nyquist are empty, so a
16 kHz clip shows its 8 kHz bandwidth directly. Features are normalised with per-band
statistics of the pool (train/dev) clips only.
"""

import os
import warnings

import torch
import torchaudio.functional as AF


_FB = {}


def raw_features(x, sr, n_mels=80, n_mfcc=40, fmax=24000, win_ms=25.0, hop_ms=10.0, fmax_mode="fixed"):
    """Log-mel [n_mels, T] and MFCC [n_mfcc, T] of a mono waveform at any sampling rate.
    The mel axis spans 0..fmax Hz for every rate; bands above the clip's Nyquist stay empty."""
    win = int(round(sr * win_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    n_fft = 1 << (win - 1).bit_length()
    top = fmax if fmax_mode == "fixed" else sr / 2
    key = (sr, n_fft, top, n_mels)
    if key not in _FB:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _FB[key] = AF.melscale_fbanks(n_fft // 2 + 1, 0.0, float(top), n_mels, sr, norm="slaney",
                                          mel_scale="slaney")
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                      center=True, return_complex=True).abs().pow(2)  # [F, T]
    mel = torch.log(_FB[key].T @ spec + 1e-8)  # [n_mels, T]
    mfcc = (mel.T @ AF.create_dct(n_mfcc, n_mels, norm="ortho")).T  # [n_mfcc, T]
    return mel, mfcc


def normalise(mel, mfcc, stats):
    return (mel - stats["mel_mu"]) / stats["mel_sd"], (mfcc - stats["mfcc_mu"]) / stats["mfcc_sd"]


class SpecCache:
    def __init__(self, cache, clips, n_mels=80, n_mfcc=40, fmax=24000, win_ms=25.0, hop_ms=10.0,
                 fmax_mode="fixed", cache_dir="data/cache"):
        self.n_mels, self.n_mfcc = n_mels, n_mfcc
        self.params = dict(n_mels=n_mels, n_mfcc=n_mfcc, fmax=fmax, win_ms=win_ms, hop_ms=hop_ms, fmax_mode=fmax_mode)
        key = f"spec_m{n_mels}_c{n_mfcc}_f{fmax}_{fmax_mode}_w{win_ms}_h{hop_ms}.pt"
        path = os.path.join(cache_dir, key) if cache_dir else None
        if path and os.path.exists(path):
            try:
                d = torch.load(path)
            except Exception:
                d = None
            if isinstance(d, dict) and set(d["mel"]) >= set(clips["clip"]):
                self.mel, self.mfcc, self.stats = d["mel"], d["mfcc"], d["stats"]
                return
        self._compute(cache, clips, n_mels, n_mfcc, fmax, win_ms, hop_ms, fmax_mode)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            torch.save({"mel": self.mel, "mfcc": self.mfcc, "stats": self.stats}, tmp)
            os.replace(tmp, path)  # atomic: concurrent jobs never read a half-written file

    def _compute(self, cache, clips, n_mels, n_mfcc, fmax, win_ms, hop_ms, fmax_mode):
        self.mel, self.mfcc = {}, {}
        for r in clips.itertuples():
            self.mel[r.clip], self.mfcc[r.clip] = raw_features(cache.native[r.clip], int(r.sr), **self.params)
        pool = clips[clips.split == "pool"]["clip"]
        m = torch.cat([self.mel[c] for c in pool], 1)
        f = torch.cat([self.mfcc[c] for c in pool], 1)
        self.stats = {"mel_mu": m.mean(1, keepdim=True), "mel_sd": m.std(1, keepdim=True) + 1e-5,
                      "mfcc_mu": f.mean(1, keepdim=True), "mfcc_sd": f.std(1, keepdim=True) + 1e-5}
        for c in self.mel:
            self.mel[c], self.mfcc[c] = normalise(self.mel[c], self.mfcc[c], self.stats)

    def batch(self, clips_in_batch, which="mel"):
        """Zero-padded [B, n, T] plus lengths."""
        store = self.mel if which == "mel" else self.mfcc
        feats = [store[c] for c in clips_in_batch]
        lens = torch.tensor([f.shape[1] for f in feats])
        out = torch.zeros(len(feats), feats[0].shape[0], int(lens.max()))
        for i, f in enumerate(feats):
            out[i, :, : f.shape[1]] = f
        return out, lens
