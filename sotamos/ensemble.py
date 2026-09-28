"""Greedy forward ensemble selection (Caruana et al. 2004) on out-of-fold dev predictions."""

from collections import Counter

import numpy as np
import pandas as pd

from sotamos.metrics import evaluate

SELECT = ("utt_LCC", "utt_SRCC", "sys_LCC", "sys_SRCC")


def sel_score(m: dict) -> float:
    return float(np.mean([m[k] for k in SELECT]))


def greedy(oof: dict, y: np.ndarray, cond: np.ndarray, max_members: int = 10, verbose: bool = True):
    """oof: name -> np.ndarray aligned with y. Returns (Counter of members, trace)."""
    score = lambda p: sel_score(evaluate(p, y, cond))  # noqa: E731
    members, total, best, trace = [], None, -1e9, []
    for _ in range(max_members):
        trial = {n: (p if total is None else (total * len(members) + p) / (len(members) + 1)) for n, p in oof.items()}
        n_best = max(trial, key=lambda n: score(trial[n]))
        s = score(trial[n_best])
        if s <= best + 1e-5:
            break
        best, total = s, trial[n_best]
        members.append(n_best)
        trace.append((n_best, s))
        if verbose:
            print(f"+ {n_best:48s} -> {s:.4f}", flush=True)
    return Counter(members), trace


def blend(preds: dict, weights: Counter) -> pd.Series:
    n = sum(weights.values())
    return sum(preds[k] * w for k, w in weights.items()) / n
