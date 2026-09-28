"""Ridge probes on frozen SSL features, scored by sentence-grouped CV on dev labels.

  uv run python scripts/probe.py data/feats/wav2vec2-large-lv60_resample16k.pt --labels both

For each layer and ridge alpha: 5-fold CV over pool sentences (the folds used by fine-tuning).
Inputs are the layer's pooled features plus one-hots for sampling rate, listening test and
their interaction. Out-of-fold predictions (mixed-test condition) are scored against dev
labels. Eval predictions average the 5 fold models. Only OOF numbers are printed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import SR_INDEX, load_clips, sentence_folds  # noqa: E402
from sotamos.metrics import evaluate  # noqa: E402

ALPHAS = [1, 10, 100, 1000, 10000]


def design(X, sr_idx, test_idx, w_onehot):
    oh = np.zeros((len(X), 3 + 2 + 6), dtype=np.float32)
    oh[np.arange(len(X)), sr_idx] = 1
    oh[np.arange(len(X)), 3 + test_idx] = 1
    oh[np.arange(len(X)), 5 + sr_idx * 2 + test_idx] = 1
    return np.concatenate([X, w_onehot * oh], 1)


def run_probe(F_pool, F_test, pool, test, fold_of, labels, alpha, w_onehot=3.0, k=5):
    sr_p = pool.sr.map(SR_INDEX).values
    sr_t = test.sr.map(SR_INDEX).values
    folds = pool.sentence.map(fold_of).values
    oof = np.zeros(len(pool))
    test_pred = np.zeros(len(test))
    for f in range(k):
        tr, va = folds != f, folds == f
        mu, sd = F_pool[tr].mean(0), F_pool[tr].std(0) + 1e-6
        Xtr = (F_pool[tr] - mu) / sd
        rows_X, rows_y = [], []
        if labels in ("single", "both"):
            rows_X.append(design(Xtr, sr_p[tr], np.zeros(tr.sum(), int), w_onehot))
            rows_y.append(pool.mos_single.values[tr])
        if labels in ("mix", "both"):
            rows_X.append(design(Xtr, sr_p[tr], np.ones(tr.sum(), int), w_onehot))
            rows_y.append(pool.mos_mix.values[tr])
        m = Ridge(alpha=alpha).fit(np.concatenate(rows_X), np.concatenate(rows_y))
        oof[va] = m.predict(design((F_pool[va] - mu) / sd, sr_p[va], np.ones(va.sum(), int), w_onehot))
        test_pred += m.predict(design((F_test - mu) / sd, sr_t, np.ones(len(test), int), w_onehot)) / k
    return oof, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--labels", default="both")
    ap.add_argument("--select", default="utt_LCC")
    ap.add_argument("--out-dir", default="results/probe")
    args = ap.parse_args()

    d = torch.load(args.feats)
    clips = load_clips().set_index("clip").loc[d["clips"]].reset_index()
    F = d["feats"].float().numpy()  # [N, L, 2D]
    is_pool = (clips.split == "pool").values
    pool, test = clips[is_pool].reset_index(drop=True), clips[~is_pool].reset_index(drop=True)
    fold_of = sentence_folds(clips, 5, 0)

    rows, best = [], None
    for layer in range(F.shape[1]):
        for alpha in ALPHAS:
            oof, tp = run_probe(F[is_pool, layer], F[~is_pool, layer], pool, test, fold_of, args.labels, alpha)
            m = evaluate(oof, pool.mos_mix.values, pool.condition.values)
            rows.append({"layer": layer, "alpha": alpha, **m})
            if best is None or m[args.select] > best[0][args.select]:
                best = (m, layer, alpha, oof, tp)
    res = pd.DataFrame(rows)
    name = Path(args.feats).stem + f"_{args.labels}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{name}_grid.csv", index=False)
    m, layer, alpha, oof, tp = best
    pd.DataFrame({"clip": pool["clip"], "pred": oof}).to_csv(out / f"{name}_oof.csv", index=False)
    pd.DataFrame({"clip": test["clip"], "pred": tp}).to_csv(out / f"{name}_test.csv", index=False)
    per_layer = res.loc[res.groupby("layer")[args.select].idxmax()]
    print(per_layer[["layer", "alpha", "utt_LCC", "utt_SRCC", "sys_SRCC", "sys_LCC"]].round(3).to_string(index=False))
    summary = {"feats": args.feats, "labels": args.labels, "best_layer": layer, "alpha": alpha,
               **{f"oof_{k}": v for k, v in m.items()}}
    json.dump(summary, open(out / f"{name}_best.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
