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


class SpecCache:
    def __init__(self, cache, clips, n_mels=80, n_mfcc=40, fmax=24000, win_ms=25.0, hop_ms=10.0,
                 fmax_mode="fixed", cache_dir="data/cache"):
        self.n_mels, self.n_mfcc = n_mels, n_mfcc
        key = f"spec_m{n_mels}_c{n_mfcc}_f{fmax}_{fmax_mode}_w{win_ms}_h{hop_ms}.pt"
        path = os.path.join(cache_dir, key) if cache_dir else None
        if path and os.path.exists(path):
            self.mel, self.mfcc = torch.load(path)
            if set(self.mel) >= set(clips["clip"]):
                return
        self._compute(cache, clips, n_mels, n_mfcc, fmax, win_ms, hop_ms, fmax_mode)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save((self.mel, self.mfcc), path)

    def _compute(self, cache, clips, n_mels, n_mfcc, fmax, win_ms, hop_ms, fmax_mode):
        self.mel, self.mfcc = {}, {}
        dct = AF.create_dct(n_mfcc, n_mels, norm="ortho")  # [n_mels, n_mfcc]
        fb_cache = {}
        for r in clips.itertuples():
            x = cache.native[r.clip]
            sr = int(r.sr)
            win = int(round(sr * win_ms / 1000))
            hop = int(round(sr * hop_ms / 1000))
            n_fft = 1 << (win - 1).bit_length()
            top = fmax if fmax_mode == "fixed" else sr / 2
            key = (sr, n_fft, top)
            if key not in fb_cache:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fb_cache[key] = AF.melscale_fbanks(n_fft // 2 + 1, 0.0, float(top), n_mels, sr, norm="slaney",
                                                       mel_scale="slaney")
            spec = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win),
                              center=True, return_complex=True).abs().pow(2)  # [F, T]
            mel = torch.log(fb_cache[key].T @ spec + 1e-8)  # [n_mels, T]
            self.mel[r.clip] = mel
            self.mfcc[r.clip] = (mel.T @ dct).T  # [n_mfcc, T]
        pool = clips[clips.split == "pool"]["clip"]
        m = torch.cat([self.mel[c] for c in pool], 1)
        f = torch.cat([self.mfcc[c] for c in pool], 1)
        self.mel_mu, self.mel_sd = m.mean(1, keepdim=True), m.std(1, keepdim=True) + 1e-5
        self.mfcc_mu, self.mfcc_sd = f.mean(1, keepdim=True), f.std(1, keepdim=True) + 1e-5
        for c in self.mel:
            self.mel[c] = (self.mel[c] - self.mel_mu) / self.mel_sd
            self.mfcc[c] = (self.mfcc[c] - self.mfcc_mu) / self.mfcc_sd

    def batch(self, clips_in_batch, which="mel"):
        """Zero-padded [B, n, T] plus lengths."""
        store = self.mel if which == "mel" else self.mfcc
        feats = [store[c] for c in clips_in_batch]
        lens = torch.tensor([f.shape[1] for f in feats])
        out = torch.zeros(len(feats), feats[0].shape[0], int(lens.max()))
        for i, f in enumerate(feats):
            out[i, :, : f.shape[1]] = f
        return out, lens
