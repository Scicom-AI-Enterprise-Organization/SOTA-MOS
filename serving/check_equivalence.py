#!/usr/bin/env python3
"""Served scores vs offline scores on the 400 eval clips, at several batch sizes.

  python check_equivalence.py ../results/final/both/system.json ../results/final/both/eval.csv

Batches mix sampling rates and lengths the way the server's batch former does (sorted by length).
Reports MAE / max difference / Pearson r against the offline predictions, and the eval metrics of
the served scores.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess_worker  # noqa: E402
from engine import Engine  # noqa: E402

from sotamos.metrics import evaluate  # noqa: E402


def main():
    system, offline_csv = sys.argv[1], sys.argv[2]
    root = Path(__file__).resolve().parents[1]
    clips = pd.read_csv(root / "data/clips.csv")
    test = clips[clips.split == "test"].reset_index(drop=True)
    offline = pd.read_csv(offline_csv).set_index("clip")["pred"].reindex(test["clip"]).values
    decoded = [preprocess_worker.decode((root / p).read_bytes()) for p in test.path]
    items = [(torch.from_numpy(n), sr, torch.from_numpy(r)) for n, sr, r, _ in decoded]
    order = np.argsort([len(r) for _, _, r in items])
    eng = Engine(system)
    res = {}
    for bs in [1, 8, 32]:
        served = np.zeros(len(items))
        for i in range(0, len(order), bs):
            idx = order[i: i + bs]
            served[idx] = eng.score([items[j] for j in idx])
        d = np.abs(served - offline)
        res[f"batch_{bs}"] = {"mae": float(d.mean()), "max": float(d.max()),
                              "pearson": float(np.corrcoef(served, offline)[0, 1]),
                              "eval": evaluate(served, test.mos_mix.values, test.condition.values)}
        print(f"batch {bs:3d}: MAE {d.mean():.2e}  max {d.max():.2e}  r {res[f'batch_{bs}']['pearson']:.6f}  "
              f"eval sys_SRCC {res[f'batch_{bs}']['eval']['sys_SRCC']:.3f} utt_LCC {res[f'batch_{bs}']['eval']['utt_LCC']:.3f}")
    json.dump(res, open(root / "results/serving_equivalence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
