"""README result figures.

  uv run python scripts/make_figures.py probes      # OOF quality per layer, frozen features
  uv run python scripts/make_figures.py ablation    # CV ablations of our model
  uv run python scripts/make_figures.py final       # eval: ours vs replication vs UTMOSv2 vs published
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos import plotstyle as ps  # noqa: E402
from sotamos.metrics import COLUMNS, PUBLISHED  # noqa: E402

ps.setup()
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

BASE = ["wav2vec2-base", "wavlm-base-plus", "hubert-base-ls960"]
LARGE = ["wav2vec2-large-lv60", "wavlm-large", "data2vec-audio-large", "hubert-large-ll60k",
         "wav2vec2-xls-r-300m", "wav2vec2-xls-r-1b"]


def probes():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, group, title in [(axes[0], BASE, "base models (12 layers)"), (axes[1], LARGE, "large models (24-48 layers)")]:
        for i, bb in enumerate(group):
            f = Path(f"results/probe/{bb}_resample16k_both_grid.csv")
            if not f.exists():
                continue
            g = pd.read_csv(f).groupby("layer").utt_LCC.max()
            depth = g.index / g.index.max()
            ax.plot(depth, g.values, color=ps.SERIES[i], lw=1.8, label=bb)
            j = int(np.argmax(g.values))
            ax.scatter([depth[j]], [g.values[j]], s=36, color=ps.SERIES[i], edgecolor=ps.SURFACE, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("relative layer depth (0 = CNN output, 1 = last layer)")
        ax.legend(loc="lower left", fontsize=7.5)
    axes[0].set_ylabel("out-of-fold utt LCC on dev (ridge probe)")
    ps.save(fig, FIG / "probe_layers.png")


def ablation():
    rows = json.load(open("results/cv_summary.json"))
    df = pd.DataFrame([{"system": r["system"], "utt_LCC": r["sel_ens"]["utt_LCC"], "sys_SRCC": r["sel_ens"]["sys_SRCC"],
                        "utt_SRCC": r["sel_ens"]["utt_SRCC"]} for r in rows]).sort_values("utt_LCC")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(df) + 1.4), sharey=True)
    y = np.arange(len(df))
    base = df.set_index("system").loc["ours_base"] if "ours_base" in set(df.system) else None
    for ax, col in zip(axes, ["utt_LCC", "sys_SRCC"]):
        colors = [ps.SERIES[1] if s == "ours_base" else ps.SERIES[0] for s in df.system]
        ax.barh(y, df[col], color=colors, height=0.62, edgecolor=ps.SURFACE)
        for yi, v in zip(y, df[col]):
            ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=7.5, color=ps.INK2)
        if base is not None:
            ax.axvline(base[col], color=ps.MUTED, lw=1, ls="--")
        lo = df[col].min() - 0.03
        ax.set_xlim(lo, min(1.0, df[col].max() + 0.03))
        ax.set_xlabel(f"out-of-fold {col.replace('_', ' ')} on dev")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df.system, fontsize=8)
    fig.suptitle("Ablations of our model (5-fold CV by sentence, dashed line = ours_base)", fontsize=11,
                 fontweight="bold", color=ps.INK)
    ps.save(fig, FIG / "ablation_oof.png")


def eval_bars(entries, fname, metric="sys_SRCC", title=None):
    """entries: list of (label, value, per_seed_values or None, kind) kind in published|replication|ours|utmos."""
    color = {"published": ps.AXIS, "replication": ps.SERIES[0], "utmos": ps.SERIES[2], "ours": ps.SERIES[1]}
    entries = sorted(entries, key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(entries) + 1.3))
    for i, (lab, v, seeds, kind) in enumerate(entries):
        ax.barh(i, v, height=0.62, color=color[kind], edgecolor=ps.SURFACE)
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color=ps.INK2)
        if seeds is not None and len(seeds) > 1:
            ax.scatter(seeds, [i] * len(seeds), s=12, color=ps.INK, zorder=3, linewidth=0)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    vals = [e[1] for e in entries] + [s for e in entries if e[2] is not None for s in e[2]]
    ax.set_xlim(max(0.0, min(vals) - 0.05), 1.0 if "MSE" not in metric else max(vals) * 1.15)
    ax.set_xlabel(f"eval {metric.replace('_', ' ')}  (bar = ensemble, dots = single seeds)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in ["published", "replication", "utmos", "ours"]]
    ax.legend(handles, ["published (challenge)", "our replication", "faster-UTMOSv2 zero-shot", "ours"],
              loc="lower right", fontsize=7.5)
    ax.set_title(title or f"Track 3 eval, {metric.replace('_', ' ')}")
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIG / fname)


def condition_scatter(pred_csv, fname, title):
    clips = pd.read_csv("data/clips.csv")
    test = clips[clips.split == "test"].set_index("clip")
    p = pd.read_csv(pred_csv).set_index("clip")["pred"]
    t = test.loc[p.index].assign(pred=p)
    g = t.groupby("condition").agg(pred=("pred", "mean"), true=("mos_mix", "mean"), sr=("sr_tag", "first"))
    srcc = stats.spearmanr(g.true, g.pred)[0]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
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
    {"probes": probes, "ablation": ablation}[sys.argv[1]]()
