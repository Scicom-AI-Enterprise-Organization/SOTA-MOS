"""MOS predictors built around a pretrained SSL encoder.

One class covers:
  - SSL-MOS (B03 baseline):  ssl -> frame head -> mean
  - HighRateMOS Model 1:      [ssl | sr_emb | multi-scale CNN(mel)] -> BLSTM -> frame head -> mean
  - HighRateMOS Model 2:      Model 1 + cross-attention (ssl queries the spectral features)
  - HighRateMOS Model 3:      Model 2 + MFCC
  - ours:                     any of the above plus a listening-test embedding, or a pooled head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel


class SSLBackbone(nn.Module):
    """layer: "last", "weighted" (softmax-weighted sum of hidden states) or an int index.
    max_layers > 0 drops every transformer block above it; the probes put the quality
    signal at 15-30% of the depth, so the upper blocks are mostly cost."""

    def __init__(self, name: str, layer="last", freeze: str = "none", max_layers: int = 0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        if max_layers:
            cfg.num_hidden_layers = max_layers  # from_pretrained then loads only the kept blocks
        # fine-tune without SpecAugment-style masking or layer drop (as mos-finetune-ssl: mask=False)
        for k, v in dict(apply_spec_augment=False, layerdrop=0.0, mask_time_prob=0.0, mask_feature_prob=0.0).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self.model = AutoModel.from_pretrained(name, config=cfg)
        try:
            self.do_normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(name), "do_normalize", False))
        except Exception:
            self.do_normalize = False
        self.dim = cfg.hidden_size
        self.n_hidden = cfg.num_hidden_layers + 1
        self.layer = layer
        if layer == "weighted":
            self.layer_w = nn.Parameter(torch.zeros(self.n_hidden))
        if freeze in ("cnn", "all"):
            for p in self.model.feature_extractor.parameters():
                p.requires_grad = False
        if freeze == "all":
            for p in self.model.parameters():
                p.requires_grad = False
        if hasattr(self.model, "masked_spec_embed"):
            self.model.masked_spec_embed = None

    def frame_lengths(self, lens):
        return self.model._get_feat_extract_output_lengths(lens).long()

    def forward(self, wav, lens):
        if self.do_normalize:
            mask = (torch.arange(wav.shape[1], device=wav.device)[None] < lens[:, None]).float()
            n = lens[:, None].float()
            mu = (wav * mask).sum(1, keepdim=True) / n
            var = (((wav - mu) * mask) ** 2).sum(1, keepdim=True) / n
            wav = (wav - mu) / torch.sqrt(var + 1e-7)
        need_all = self.layer != "last"
        out = self.model(wav, output_hidden_states=need_all)
        if self.layer == "last":
            h = out.last_hidden_state
        elif self.layer == "weighted":
            w = torch.softmax(self.layer_w, 0)
            h = sum(w[i] * out.hidden_states[i] for i in range(self.n_hidden))
        else:
            h = out.hidden_states[int(self.layer)]
        flens = self.frame_lengths(lens).clamp(max=h.shape[1])
        return h, flens


class MultiScaleCNN(nn.Module):
    """Parallel 2-D conv stacks with 3x3 / 5x5 / 7x7 kernels over the (mel, time) plane.
    Frequency is pooled to `f_bins` coarse bands, so band position (bandwidth) is kept."""

    def __init__(self, ch=32, out_dim=128, kernels=(3, 5, 7), f_bins=8):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, ch, k, padding=k // 2), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, k, padding=k // 2), nn.BatchNorm2d(ch), nn.ReLU(),
            )
            for k in kernels
        )
        self.f_bins = f_bins
        self.proj = nn.Linear(ch * len(kernels) * f_bins, out_dim)

    def forward(self, x):  # x: [B, n_mels, T]
        x = x.unsqueeze(1)
        y = torch.cat([b(x) for b in self.branches], 1)  # [B, C, n_mels, T]
        y = F.adaptive_avg_pool2d(y, (self.f_bins, y.shape[-1]))  # [B, C, f_bins, T]
        y = y.flatten(1, 2).transpose(1, 2)  # [B, T, C*f_bins]
        return self.proj(y)


def align_frames(feat, feat_lens, frame_lens, n_frames):
    """feat [B, T, D] at the spectral frame rate -> [B, n_frames, D] on the SSL frame grid.
    Each item's valid part is resampled to its own SSL length, then tiled like the waveform."""
    out = feat.new_zeros(feat.shape[0], n_frames, feat.shape[2])
    for i in range(feat.shape[0]):
        f = feat[i, : int(feat_lens[i])].T.unsqueeze(0)  # [1, D, T_i]
        n = max(int(frame_lens[i]), 1)
        f = F.interpolate(f, size=n, mode="linear", align_corners=False)[0].T  # [n, D]
        reps = (n_frames + n - 1) // n
        out[i] = f.repeat(reps, 1)[:n_frames]
    return out


def masked_mean(x, lens):
    mask = (torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]).to(x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(1) / mask.sum(1).clamp(min=1)


class FrameHead(nn.Module):
    """SHEET/SSL-MOS projection: Linear -> ReLU -> Dropout(0.3) -> Linear -> tanh*2+3, per frame."""

    def __init__(self, d_in, hidden=64, range_clip=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))
        self.range_clip = range_clip

    def forward(self, x):
        y = self.net(x).squeeze(-1)
        return torch.tanh(y) * 2.0 + 3.0 if self.range_clip else y


class MOSModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.ssl = SSLBackbone(cfg["backbone"], cfg.get("layer", "last"), cfg.get("freeze", "none"),
                               cfg.get("max_layers", 0))
        d_proj = cfg.get("d_ssl_proj", 0)
        self.ssl_proj = nn.Linear(self.ssl.dim, d_proj) if d_proj else nn.Identity()
        d_frame = d_proj or self.ssl.dim

        self.d_sr = cfg.get("sr_emb", 0)
        self.sr_emb = nn.Embedding(3, self.d_sr) if self.d_sr else None
        self.d_test = cfg.get("test_emb", 0)
        self.test_emb = nn.Embedding(2, self.d_test) if self.d_test else None
        self.use_mel = cfg.get("mel", False)
        self.use_mfcc = cfg.get("mfcc", False)
        self.use_xattn = cfg.get("xattn", False)
        head = cfg.get("head", "frame")

        d_spec = 0
        if self.use_mel:
            self.mel_cnn = MultiScaleCNN(cfg.get("mel_ch", 32), cfg.get("d_mel", 128), f_bins=cfg.get("mel_fbins", 8))
            d_spec += cfg.get("d_mel", 128)
        if self.use_mfcc:
            self.mfcc_proj = nn.Sequential(nn.Linear(cfg.get("n_mfcc", 40), cfg.get("d_mfcc", 64)), nn.ReLU())
            d_spec += cfg.get("d_mfcc", 64)
        d_fused = d_frame + d_spec
        if self.use_xattn:
            assert d_spec > 0, "cross-attention needs spectral features"
            d_att = cfg.get("d_xattn", 256)
            self.q_proj = nn.Linear(d_frame, d_att)
            self.kv_proj = nn.Linear(d_spec, d_att)
            self.xattn = nn.MultiheadAttention(d_att, cfg.get("xattn_heads", 4), batch_first=True, dropout=0.1)
            d_fused += d_att
        # sr / test embeddings are broadcast along time and concatenated (HighRateMOS Model 1)
        cond_on_frames = head == "frame"
        if cond_on_frames:
            d_fused += self.d_sr + self.d_test
        self.use_blstm = cfg.get("blstm", False)
        if self.use_blstm:
            h = cfg.get("blstm_hidden", 128)
            self.blstm = nn.LSTM(d_fused, h, batch_first=True, bidirectional=True)
            d_fused = 2 * h
        self.head_type = head
        if head == "frame":
            self.head = FrameHead(d_fused, cfg.get("head_hidden", 64), cfg.get("range_clip", True))
        elif head == "pool":
            d_pool = 2 * d_fused + self.d_sr + self.d_test
            hid = cfg.get("head_hidden", 256)
            self.head = nn.Sequential(nn.Linear(d_pool, hid), nn.GELU(), nn.Dropout(cfg.get("head_dropout", 0.2)),
                                      nn.Linear(hid, 1))
        else:
            raise ValueError(head)

    def forward(self, batch):
        wav, lens = batch["wav"], batch["lens"]
        h, flens = self.ssl(wav, lens)
        h = self.ssl_proj(h)
        B, N, _ = h.shape
        parts = [h]
        spec_parts = []
        if self.use_mel:
            m = self.mel_cnn(batch["mel"])  # [B, Tm, d]
            spec_parts.append(align_frames(m, batch["mel_lens"], flens, N))
        if self.use_mfcc:
            m = self.mfcc_proj(batch["mfcc"].transpose(1, 2))  # [B, Tm, d]
            spec_parts.append(align_frames(m, batch["mfcc_lens"], flens, N))
        if spec_parts:
            spec = torch.cat(spec_parts, -1)
            parts.append(spec)
            if self.use_xattn:
                q, kv = self.q_proj(h), self.kv_proj(spec)
                pad = torch.arange(N, device=h.device)[None] >= flens[:, None]
                att, _ = self.xattn(q, kv, kv, key_padding_mask=pad)
                parts.append(att)
        cond = []
        if self.sr_emb is not None:
            cond.append(self.sr_emb(batch["sr_idx"]))
        if self.test_emb is not None:
            cond.append(self.test_emb(batch["test_idx"]))
        if self.head_type == "frame":
            parts += [c[:, None, :].expand(B, N, c.shape[-1]) for c in cond]
        x = torch.cat(parts, -1)
        if self.use_blstm:
            x, _ = self.blstm(x)
        if self.head_type == "frame":
            frame_scores = self.head(x)  # [B, N]
            score = masked_mean(frame_scores, flens)
            return {"score": score, "frame_scores": frame_scores, "frame_lens": flens}
        mu = masked_mean(x, flens)
        mask = (torch.arange(N, device=x.device)[None] < flens[:, None]).to(x.dtype).unsqueeze(-1)
        sd = torch.sqrt(((x - mu[:, None]) ** 2 * mask).sum(1) / mask.sum(1).clamp(min=1) + 1e-5)
        z = torch.cat([mu, sd] + cond, -1)
        y = self.head(z).squeeze(-1)
        if self.cfg.get("range_clip", True):
            y = torch.tanh(y) * 2.0 + 3.0
        return {"score": y, "frame_scores": None, "frame_lens": flens}
