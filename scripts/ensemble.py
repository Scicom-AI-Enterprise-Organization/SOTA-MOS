"""Greedy forward ensemble selection on out-of-fold dev predictions, then one eval score.

  uv run python scripts/ensemble.py NAME=OOF_CSV:EVAL_CSV [...] --max-members 10

Candidates are averaged with replacement (Caruana et al. 2004). Each step adds the candidate
that most improves the selection score on the pool (dev labels, out-of-fold predictions).
Eval metrics are computed only for the final selection.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.metrics import evaluate  # noqa: E402

SELECT = ("utt_LCC", "utt_SRCC", "sys_LCC", "sys_SRCC")


def sel_score(m):
    return float(np.mean([m[k] for k in SELECT]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--max-members", type=int, default=10)
    ap.add_argument("--out", default="results/ensemble")
    args = ap.parse_args()
    clips = pd.read_csv("data/clips.csv").set_index("clip")
    pool = clips[clips.split == "pool"]
    test = clips[clips.split == "test"]
    oof, ev = {}, {}
    for c in args.candidates:
        name, paths = c.split("=", 1)
        o, e = paths.split(":")
        oof[name] = pd.read_csv(o).set_index("clip")["pred"].reindex(pool.index)
        ev[name] = pd.read_csv(e).set_index("clip")["pred"].reindex(test.index)
        assert oof[name].notna().all() and ev[name].notna().all(), name
    score = lambda p: evaluate(p.values, pool.mos_mix.values, pool.condition.values)  # noqa: E731
    solo = {n: score(p) for n, p in oof.items()}
    for n, m in sorted(solo.items(), key=lambda kv: -sel_score(kv[1])):
        print(f"  {sel_score(m):.4f}  utt_LCC {m['utt_LCC']:.3f} sys_SRCC {m['sys_SRCC']:.3f}  {n}")
    members, best = [], -1e9
    total = None
    for _ in range(args.max_members):
        trial = {n: (p if total is None else (total * len(members) + p) / (len(members) + 1)) for n, p in oof.items()}
        n_best = max(trial, key=lambda n: sel_score(score(trial[n])))
        s = sel_score(score(trial[n_best]))
        if s <= best + 1e-5:
            break
        best, total = s, trial[n_best]
        members.append(n_best)
        print(f"+ {n_best:40s} -> {s:.4f}")
    w = Counter(members)
    e = sum(ev[n] * k for n, k in w.items()) / len(members)
    o = sum(oof[n] * k for n, k in w.items()) / len(members)
    res = {"members": dict(w), "oof": score(o), "eval": evaluate(e.values, test.mos_mix.values, test.condition.values)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    e.rename("pred").to_csv(Path(args.out) / "eval.csv")
    o.rename("pred").to_csv(Path(args.out) / "oof.csv")
    json.dump(res, open(Path(args.out) / "result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
