"""README figures from results/*_summary.json and prediction files.

  uv run python scripts/plot_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos import plotstyle as ps  # noqa: E402
from sotamos.metrics import PUBLISHED  # noqa: E402

ps.setup()
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)


def load(group):
    p = Path(f"results/{group}_summary.json")
    return json.load(open(p)) if p.exists() else []


def bars_vs_published(entries, fname, title, metric="sys_SRCC"):
    """Horizontal bars: published Track 3 systems (grey) and ours (blue), per-seed dots on ours."""
    rows = [(n, r[metric], None, "published") for n, r in PUBLISHED.iterrows()]
    for e in entries:
        seeds = [s[f"eval_{metric}"] for s in e["per_seed"]]
        rows.append((e["label"], e["eval_ens"][metric], seeds, "ours"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.2))
    y = np.arange(len(rows))
    for i, (name, v, seeds, kind) in enumerate(rows):
        col = ps.SERIES[0] if kind == "ours" else ps.AXIS
        ax.barh(i, v, height=0.62, color=col, edgecolor=ps.SURFACE)
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=ps.INK2)
        if seeds and len(seeds) > 1:
            ax.scatter(seeds, [i] * len(seeds), s=14, color=ps.INK, zorder=3, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    lo = min(min(r[1] for r in rows), min(min(r[2]) for r in rows if r[2])) - 0.05
    ax.set_xlim(max(0, lo), 1.0)
    ax.set_xlabel(f"eval {metric.replace('_', ' ')}  (bar = seed ensemble, dots = single seeds)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIG / fname)


def condition_scatter(pred_csv, fname, title):
    clips = pd.read_csv("data/clips.csv")
    test = clips[clips.split == "test"].set_index("clip")
    p = pd.read_csv(pred_csv).set_index("clip")["pred"]
    t = test.loc[p.index].assign(pred=p)
    g = t.groupby("condition").agg(pred=("pred", "mean"), true=("mos_mix", "mean"), sr=("sr_tag", "first"))
    from scipy import stats
    srcc = stats.spearmanr(g.true, g.pred)[0]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo, hi = 1.5, 4.5
    ax.plot([lo, hi], [lo, hi], color=ps.AXIS, lw=1, ls="--")
    for sr in ["16k", "24k", "48k"]:
        s = g[g.sr == sr]
        ax.scatter(s.true, s.pred, s=46, color=ps.SR_COLOR[sr], marker=ps.SR_MARKER[sr], edgecolor=ps.SURFACE,
                   linewidth=1.2, label=sr, zorder=3)
    for c, r in g.iterrows():
        ax.annotate(c, (r.true, r.pred), xytext=(4, -3), textcoords="offset points", fontsize=6.5, color=ps.INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true condition MOS (eval)")
    ax.set_ylabel("predicted condition MOS")
    ax.set_title(f"{title}: sys SRCC {srcc:.3f}")
    ax.legend(loc="upper left")
    ps.save(fig, FIG / fname)


if __name__ == "__main__":
    print("import-only; figures are built by scripts/make_figures.py once results exist")
