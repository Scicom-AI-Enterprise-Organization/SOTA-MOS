"""Build the final systems with the pre-registered rule, then score them on eval once.

  uv run python scripts/final.py              # select + score eval once, at the very end
  uv run python scripts/final.py --no-eval    # progress check: out-of-fold only

Candidates are anything with out-of-fold predictions on the 400 dev clips plus eval predictions:
  - CV systems under exp/cv/ (5 folds, seeds averaged)
  - frozen-feature probes under results/probe/ and results/probe_krr/
They split by the labels they trained on:
  train+dev   labels both or mix (uses the dev labels released in the evaluation phase)
  train-only  labels single (the HighRateMOS setting)
Selection: greedy forward selection with replacement on mean(utt_LCC, utt_SRCC, sys_LCC,
sys_SRCC) of the out-of-fold predictions against dev labels, at most 10 members.
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.ensemble import blend, greedy, sel_score  # noqa: E402
from sotamos.metrics import evaluate  # noqa: E402
from sotamos.report import POOL, TEST, system_predictions  # noqa: E402

OUT = Path("results/final")


def candidates():
    cands = {"both": {}, "single": {}}
    for d in sorted(Path("exp/cv").iterdir()):
        preds = system_predictions(d)
        if not preds:
            continue
        cfg_labels = next(iter(preds.values()))["folds"][0]["cfg"]["labels"]
        group = "single" if cfg_labels == "single" else "both"
        oof = pd.concat([p["sel"] for p in preds.values()], axis=1).mean(1)
        test = pd.concat([p["test"] for p in preds.values()], axis=1).mean(1)
        cands[group][f"ft:{d.name}"] = (oof, test)
    for pattern, kind in [("results/probe/*_oof.csv", "ridge"), ("results/probe_krr/*_oof.csv", "krr")]:
        for f in sorted(glob.glob(pattern)):
            name = Path(f).name.removesuffix("_oof.csv")
            if "_both" in name:
                group = "both"
            elif "_single" in name:
                group = "single"
            else:
                continue
            oof = pd.read_csv(f).set_index("clip")["pred"]
            test = pd.read_csv(f.replace("_oof.csv", "_test.csv")).set_index("clip")["pred"]
            cands[group][f"{kind}:{name}"] = (oof, test)
    return cands


def main():
    no_eval = "--no-eval" in sys.argv  # progress checks: out-of-fold only, nothing touches eval labels
    OUT.mkdir(parents=True, exist_ok=True)
    y, cond = POOL.mos_mix, POOL.condition
    summary = {}
    for group, label in [("both", "train+dev"), ("single", "train-only")]:
        c = candidates()[group]
        oof = {n: v[0].reindex(POOL.index).values for n, v in c.items()}
        test = {n: v[1].reindex(TEST.index) for n, v in c.items()}
        print(f"\n=== {label}: {len(c)} candidates")
        solo = sorted(((sel_score(evaluate(p, y.values, cond.values)), n) for n, p in oof.items()), reverse=True)
        for s, n in solo[:12]:
            print(f"  {s:.4f}  {n}")
        weights, trace = greedy(oof, y.values, cond.values, max_members=10)
        o = pd.Series(blend(oof, weights), index=POOL.index)
        print(f"members {dict(weights)}")
        print(f"OOF  {json.dumps({k: round(v, 3) for k, v in evaluate(o.values, y.values, cond.values).items()})}")
        if no_eval:
            continue
        e = blend(test, weights)
        res = {"label": label, "members": dict(weights), "trace": trace,
               "oof": evaluate(o.values, y.values, cond.values),
               "eval": evaluate(e.values, TEST.mos_mix.values, TEST.condition.values)}
        d = OUT / group
        d.mkdir(exist_ok=True)
        o.rename("pred").to_csv(d / "oof.csv")
        e.rename("pred").to_csv(d / "eval.csv")
        json.dump(res, open(d / "result.json", "w"), indent=1)
        summary[label] = res
        print(f"EVAL {json.dumps({k: round(v, 3) for k, v in res['eval'].items()})}")
    if not no_eval:
        json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
