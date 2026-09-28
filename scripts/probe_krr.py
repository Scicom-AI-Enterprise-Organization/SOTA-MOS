"""Kernel-ridge probes on the best frozen layer: RBF kernel on SSL features + linear kernel on
sampling-rate / listening-test one-hots, so the per-rate offsets stay learnable.

  uv run python scripts/probe_krr.py data/feats/hubert-large-ll60k_resample16k.pt --labels both

Same sentence folds and scoring as scripts/probe.py (out-of-fold on dev labels only). The layer
comes from the ridge probe's *_best.json; `--layers k` averages the k layers around it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import SR_INDEX, load_clips, sentence_folds  # noqa: E402
from sotamos.metrics import evaluate  # noqa: E402

ALPHAS = [0.03, 0.1, 0.3, 1.0]
GAMMAS = [0.25, 0.5, 1.0, 2.0]  # x 1/n_features


def onehot(sr_idx, test_idx):
    oh = np.zeros((len(sr_idx), 11), dtype=np.float64)
    oh[np.arange(len(sr_idx)), sr_idx] = 1
    oh[np.arange(len(sr_idx)), 3 + test_idx] = 1
    oh[np.arange(len(sr_idx)), 5 + sr_idx * 2 + test_idx] = 1
    return oh


def kernel(A, B, oa, ob, gamma, c=1.0):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0)) + c * oa @ ob.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--labels", default="both")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--out-dir", default="results/probe_krr")
    args = ap.parse_args()
    d = torch.load(args.feats)
    clips = load_clips().set_index("clip").loc[d["clips"]].reset_index()
    stem = Path(args.feats).stem
    best = json.load(open(f"results/probe/{stem}_{args.labels}_best.json"))["best_layer"]
    lo = max(0, best - args.layers // 2)
    F = d["feats"][:, lo: lo + args.layers].float().mean(1).numpy().astype(np.float64)
    is_pool = (clips.split == "pool").values
    pool, test = clips[is_pool].reset_index(drop=True), clips[~is_pool].reset_index(drop=True)
    Fp, Ft = F[is_pool], F[~is_pool]
    folds = pool.sentence.map(sentence_folds(clips, 5, 0)).values
    sr_p, sr_t = pool.sr.map(SR_INDEX).values, test.sr.map(SR_INDEX).values
    results = []
    for alpha in ALPHAS:
        for g in GAMMAS:
            gamma = g / Fp.shape[1]
            oof, tp = np.zeros(len(pool)), np.zeros(len(test))
            for f in range(5):
                tr, va = folds != f, folds == f
                mu, sd = Fp[tr].mean(0), Fp[tr].std(0) + 1e-6
                X = (Fp[tr] - mu) / sd
                Xs, ys, os_ = [], [], []
                if args.labels in ("single", "both"):
                    Xs.append(X); ys.append(pool.mos_single.values[tr]); os_.append(onehot(sr_p[tr], np.zeros(tr.sum(), int)))
                if args.labels in ("mix", "both"):
                    Xs.append(X); ys.append(pool.mos_mix.values[tr]); os_.append(onehot(sr_p[tr], np.ones(tr.sum(), int)))
                X, y, O = np.concatenate(Xs), np.concatenate(ys), np.concatenate(os_)
                ym = y.mean()
                m = KernelRidge(alpha=alpha, kernel="precomputed").fit(kernel(X, X, O, O, gamma), y - ym)
                Xv = (Fp[va] - mu) / sd
                oof[va] = m.predict(kernel(Xv, X, onehot(sr_p[va], np.ones(va.sum(), int)), O, gamma)) + ym
                Xt = (Ft - mu) / sd
                tp += (m.predict(kernel(Xt, X, onehot(sr_t, np.ones(len(test), int)), O, gamma)) + ym) / 5
            met = evaluate(oof, pool.mos_mix.values, pool.condition.values)
            results.append((met, alpha, g, oof, tp))
    met, alpha, g, oof, tp = max(results, key=lambda r: r[0]["utt_LCC"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{args.labels}_L{args.layers}"
    pd.DataFrame({"clip": pool["clip"], "pred": oof}).to_csv(out / f"{name}_oof.csv", index=False)
    pd.DataFrame({"clip": test["clip"], "pred": tp}).to_csv(out / f"{name}_test.csv", index=False)
    summary = {"feats": args.feats, "labels": args.labels, "layer": best, "layers": args.layers, "alpha": alpha,
               "gamma_scale": g, **{f"oof_{k}": v for k, v in met.items()}}
    json.dump(summary, open(out / f"{name}_best.json", "w"), indent=1)
    print(name, {k: round(v, 3) for k, v in met.items()}, "alpha", alpha, "gamma", g)


if __name__ == "__main__":
    main()
