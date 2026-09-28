"""Frozen-feature probe members (ridge, kernel ridge): fit per fold, save, and predict new audio.

A probe member = (backbone, input mode, layer window, labels, model kind, hyperparameters).
fit_member() refits it on the same 5 sentence folds as scripts/probe.py / probe_krr.py, so its
out-of-fold and eval predictions reproduce the saved ones. predict() averages the fold models.
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge

from sotamos.data import SR_INDEX, sentence_folds


def design_onehot(sr_idx, test_idx):
    oh = np.zeros((len(sr_idx), 11), dtype=np.float64)
    oh[np.arange(len(sr_idx)), sr_idx] = 1
    oh[np.arange(len(sr_idx)), 3 + test_idx] = 1
    oh[np.arange(len(sr_idx)), 5 + sr_idx * 2 + test_idx] = 1
    return oh


def rbf_plus_linear(A, B, oa, ob, gamma, c=1.0):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0)) + c * oa @ ob.T


def layer_features(F, layer, window=1):
    lo = max(0, layer - window // 2)
    return F[:, lo: lo + window].float().mean(1).numpy().astype(np.float64)


def fit_member(spec, F_pool, pool, clips, k=5):
    """spec: dict(kind, labels, alpha, [gamma_scale], [w_onehot]). Returns list of fold states."""
    sr = pool.sr.map(SR_INDEX).values
    folds = pool.sentence.map(sentence_folds(clips, k, 0)).values
    states = []
    for f in range(k):
        tr = folds != f
        mu, sd = F_pool[tr].mean(0), F_pool[tr].std(0) + 1e-6
        X = (F_pool[tr] - mu) / sd
        Xs, ys, os_ = [], [], []
        if spec["labels"] in ("single", "both"):
            Xs.append(X); ys.append(pool.mos_single.values[tr]); os_.append(design_onehot(sr[tr], np.zeros(tr.sum(), int)))
        if spec["labels"] in ("mix", "both"):
            Xs.append(X); ys.append(pool.mos_mix.values[tr]); os_.append(design_onehot(sr[tr], np.ones(tr.sum(), int)))
        X, y, O = np.concatenate(Xs), np.concatenate(ys), np.concatenate(os_)
        st = {"mu": mu, "sd": sd, "fold": f}
        if spec["kind"] == "ridge":
            w = spec.get("w_onehot", 3.0)
            m = Ridge(alpha=spec["alpha"]).fit(np.concatenate([X, w * O], 1), y)
            st.update(coef=m.coef_, intercept=m.intercept_, w_onehot=w)
        else:
            from sklearn.kernel_ridge import KernelRidge

            gamma = spec["gamma_scale"] / X.shape[1]
            ym = y.mean()
            m = KernelRidge(alpha=spec["alpha"], kernel="precomputed").fit(rbf_plus_linear(X, X, O, O, gamma), y - ym)
            st.update(X=X, O=O, dual=m.dual_coef_, ym=ym, gamma=gamma)
        states.append(st)
    return states


def predict_fold(st, kind, F, sr_idx, test_name="mix"):
    t = np.full(len(F), 1 if test_name == "mix" else 0)
    O = design_onehot(sr_idx, t)
    X = (F - st["mu"]) / st["sd"]
    if kind == "ridge":
        return np.concatenate([X, st["w_onehot"] * O], 1) @ st["coef"] + st["intercept"]
    return rbf_plus_linear(X, st["X"], O, st["O"], st["gamma"]) @ st["dual"] + st["ym"]


def oof_and_test(states, kind, F_pool, F_test, pool, test, clips, labels, k=5):
    folds = pool.sentence.map(sentence_folds(clips, k, 0)).values
    sr_p, sr_t = pool.sr.map(SR_INDEX).values, test.sr.map(SR_INDEX).values
    oof, tp = np.zeros(len(pool)), np.zeros(len(test))
    for st in states:
        va = folds == st["fold"]
        oof[va] = predict_fold(st, kind, F_pool[va], sr_p[va])
        tp += predict_fold(st, kind, F_test, sr_t) / len(states)
    return oof, tp


@torch.no_grad()
def pooled_hidden_states(ssl, wav, device):
    """Mean and std over frames of every hidden layer: [L, 2D] (as scripts/extract_ssl_features.py)."""
    x = wav[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        if ssl.do_normalize:
            x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        hs = ssl.model(x, output_hidden_states=True).hidden_states
    h = torch.stack([t[0].float() for t in hs])
    return torch.cat([h.mean(1), h.std(1)], -1).half().cpu()
