"""Losses. Config form: {"name": weight, ...} with optional params in `loss_params`.

  l1 / l2               clipped utterance-level L1/L2 (errors below tau are ignored)
  l1_frame / l2_frame   clipped frame-level L1/L2 against the broadcast utterance target (SHEET)
  contrastive           UTMOS pairwise loss: |(p_i - p_j) - (y_i - y_j)| beyond a margin
  lcc                   1 - Pearson correlation over the batch
  ccc                   1 - concordance correlation over the batch
"""

import torch
import torch.nn.functional as F


def _clipped(err, tau, order):
    loss = err.abs() if order == 1 else err ** 2
    return (loss * (err.abs() > tau)).mean()


def contrastive(pred, target, margin=0.1):
    gd = target[:, None] - target[None, :]
    pd = pred[:, None] - pred[None, :]
    return F.relu((pd - gd).abs() - margin).mean() / 2


def lcc_loss(pred, target):
    p = pred - pred.mean()
    t = target - target.mean()
    return 1 - (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def ccc_loss(pred, target):
    pm, tm = pred.mean(), target.mean()
    pv, tv = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean()
    return 1 - 2 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)


class Criterion:
    def __init__(self, weights: dict, params: dict | None = None):
        self.w = weights
        self.p = params or {}

    def __call__(self, out, target):
        pred = out["score"].float()
        target = target.float()
        terms = {}
        tau = self.p.get("tau", 0.25)
        for name, w in self.w.items():
            if not w:
                continue
            if name in ("l1", "l2"):
                terms[name] = _clipped(pred - target, tau, 1 if name == "l1" else 2)
            elif name in ("l1_frame", "l2_frame"):
                fs, fl = out["frame_scores"].float(), out["frame_lens"]
                mask = torch.arange(fs.shape[1], device=fs.device)[None] < fl[:, None]
                err = (fs - target[:, None])[mask]
                terms[name] = _clipped(err, tau, 1 if name == "l1_frame" else 2)
            elif name == "contrastive":
                terms[name] = contrastive(pred, target, self.p.get("margin", 0.1))
            elif name == "lcc":
                terms[name] = lcc_loss(pred, target)
            elif name == "ccc":
                terms[name] = ccc_loss(pred, target)
            else:
                raise ValueError(name)
        total = sum(self.w[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
