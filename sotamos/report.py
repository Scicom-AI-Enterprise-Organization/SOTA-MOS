"""Collect runs, compute dev / out-of-fold and eval metrics, build ensembles.

Run layout: exp/<group>/<system>/<run>, run = s{seed} (dev/full protocol) or s{seed}_f{fold} (cv).
Selection-side numbers ("dev" / "oof") use only pool clips and dev (mixed-test Part 1) labels.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sotamos.metrics import COLUMNS, evaluate

RUN_RE = re.compile(r"s(\d+)(?:_f(\d+))?$")


def load_clips():
    c = pd.read_csv("data/clips.csv")
    return c[c.split == "pool"].set_index("clip"), c[c.split == "test"].set_index("clip")


POOL, TEST = load_clips()


def _read(path):
    return pd.read_csv(path).set_index("clip")["pred"] if path.exists() else None


def load_runs(system_dir):
    runs = []
    for d in sorted(Path(system_dir).glob("s*")):
        m = RUN_RE.match(d.name)
        if not m or not (d / "done").exists():
            continue
        runs.append(dict(
            name=d.name, seed=int(m.group(1)), fold=int(m.group(2)) if m.group(2) else None,
            cfg=yaml.safe_load(open(d / "config.yaml")), val=_read(d / "pred_val.csv"),
            test=_read(d / "pred_test.csv"), pool=_read(d / "pred_pool.csv"),
            result=json.load(open(d / "result.json")),
        ))
    return runs


def score_dev(pred: pd.Series, target="mos_mix"):
    p = POOL.loc[pred.index]
    return evaluate(pred.values, p[target].values, p.condition.values)


def score_eval(pred: pd.Series):
    t = TEST.loc[pred.index]
    return evaluate(pred.values, t.mos_mix.values, t.condition.values)


def system_predictions(system_dir, k_folds=5):
    """Per seed: selection-side pool predictions (dev preds or OOF) and eval predictions."""
    runs = load_runs(system_dir)
    if not runs:
        return {}
    out = {}
    for seed in sorted({r["seed"] for r in runs}):
        rs = [r for r in runs if r["seed"] == seed]
        if rs[0]["fold"] is None:
            r = rs[0]
            out[seed] = {"sel": r["val"], "test": r["test"], "kind": "dev"}
        else:
            if len(rs) < k_folds:
                continue
            oof = pd.concat([r["val"] for r in rs])
            test = pd.concat([r["test"] for r in rs], axis=1).mean(1)
            out[seed] = {"sel": oof, "test": test, "kind": "oof", "folds": rs}
    return out


def summarize(system_dir, label=None):
    preds = system_predictions(system_dir)
    if not preds:
        return None
    rows = []
    for seed, p in preds.items():
        rows.append({"seed": seed, **{f"sel_{k}": v for k, v in score_dev(p["sel"]).items()},
                     **{f"eval_{k}": v for k, v in score_eval(p["test"]).items()}})
    per_seed = pd.DataFrame(rows)
    sel_ens = pd.concat([p["sel"] for p in preds.values()], axis=1).mean(1)
    test_ens = pd.concat([p["test"] for p in preds.values()], axis=1).mean(1)
    return {
        "system": label or Path(system_dir).name,
        "kind": next(iter(preds.values()))["kind"],
        "n_seeds": len(preds),
        "per_seed": per_seed,
        "sel_ens": score_dev(sel_ens),
        "eval_ens": score_eval(test_ens),
        "sel_pred": sel_ens,
        "test_pred": test_ens,
    }


def highratemos_ensemble(rep_dir="exp/rep", suffix=""):
    """HighRateMOS (paper Sec. V-D): Model 1 best among 5 folds + Models 2 and 3 trained on the full
    training set. The best fold is the one with the highest dev system-level SRCC."""
    cv = load_runs(f"{rep_dir}/hrm_model1_cv{suffix}")
    m2 = {r["seed"]: r for r in load_runs(f"{rep_dir}/hrm_model2{suffix}")}
    m3 = {r["seed"]: r for r in load_runs(f"{rep_dir}/hrm_model3{suffix}")}
    rows, sel_preds, test_preds = [], [], []
    for seed in sorted({r["seed"] for r in cv}):
        folds = [r for r in cv if r["seed"] == seed]
        if len(folds) < 5 or seed not in m2 or seed not in m3:
            continue
        best = max(folds, key=lambda r: score_dev(r["pool"])["sys_SRCC"])
        sel = pd.concat([best["pool"], m2[seed]["val"], m3[seed]["val"]], axis=1).mean(1)
        test = pd.concat([best["test"], m2[seed]["test"], m3[seed]["test"]], axis=1).mean(1)
        sel_preds.append(sel)
        test_preds.append(test)
        rows.append({"seed": seed, "best_fold": best["fold"],
                     **{f"sel_{k}": v for k, v in score_dev(sel).items()},
                     **{f"eval_{k}": v for k, v in score_eval(test).items()}})
    if not rows:
        return None
    sel_ens = pd.concat(sel_preds, axis=1).mean(1)
    test_ens = pd.concat(test_preds, axis=1).mean(1)
    return {"system": "HighRateMOS ensemble (ours)" + suffix, "kind": "dev", "n_seeds": len(rows),
            "per_seed": pd.DataFrame(rows), "sel_ens": score_dev(sel_ens), "eval_ens": score_eval(test_ens),
            "sel_pred": sel_ens, "test_pred": test_ens}


def fmt_table(summaries, published=None, sel_cols=("sys_SRCC", "utt_LCC")):
    """Markdown table: selection-side columns, then eval (8 official metrics) for the seed ensemble,
    plus the per-seed mean +- std of eval sys_SRCC."""
    head = ["system", "seeds"] + [f"sel {c}" for c in sel_cols] + [c.replace("_", " ") for c in COLUMNS] + \
           ["sys SRCC per seed"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        if s is None:
            continue
        ps = s["per_seed"]["eval_sys_SRCC"]
        cells = [s["system"], str(s["n_seeds"])] + [f"{s['sel_ens'][c]:.3f}" for c in sel_cols] + \
                [f"{s['eval_ens'][c]:.3f}" for c in COLUMNS] + [f"{ps.mean():.3f} ± {ps.std(ddof=0):.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    if published is not None:
        for name, r in published.iterrows():
            cells = [f"*{name} (published)*", "–"] + ["–"] * len(sel_cols) + [f"{r[c]:.3f}" for c in COLUMNS] + ["–"]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_jsonable(s):
    return {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
            for k, v in s.items() if k not in ("sel_pred", "test_pred")}


def np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
