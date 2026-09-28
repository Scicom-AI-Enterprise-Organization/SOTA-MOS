"""Greedy forward ensemble selection on out-of-fold dev predictions for an ad-hoc candidate list.

  uv run python scripts/ensemble.py NAME=OOF_CSV:EVAL_CSV [...] --max-members 10 [--no-eval]

scripts/final.py applies the same selection to every candidate; use this for quick checks.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.ensemble import blend, greedy, sel_score  # noqa: E402
from sotamos.metrics import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--max-members", type=int, default=10)
    ap.add_argument("--out", default="results/ensemble")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    clips = pd.read_csv("data/clips.csv").set_index("clip")
    pool, test = clips[clips.split == "pool"], clips[clips.split == "test"]
    oof, ev = {}, {}
    for c in args.candidates:
        name, paths = c.split("=", 1)
        o, e = paths.split(":")
        oof[name] = pd.read_csv(o).set_index("clip")["pred"].reindex(pool.index).values
        ev[name] = pd.read_csv(e).set_index("clip")["pred"].reindex(test.index)
    y, cond = pool.mos_mix.values, pool.condition.values
    for n in sorted(oof, key=lambda n: -sel_score(evaluate(oof[n], y, cond))):
        m = evaluate(oof[n], y, cond)
        print(f"  {sel_score(m):.4f}  utt_LCC {m['utt_LCC']:.3f} sys_SRCC {m['sys_SRCC']:.3f}  {n}")
    weights, _ = greedy(oof, y, cond, args.max_members)
    res = {"members": dict(weights), "oof": evaluate(blend(oof, weights), y, cond)}
    if not args.no_eval:
        e = blend(ev, weights)
        res["eval"] = evaluate(e.values, test.mos_mix.values, test.condition.values)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        e.rename("pred").to_csv(Path(args.out) / "eval.csv")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(args.out) / "result.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "eval"}, indent=1))


if __name__ == "__main__":
    main()
