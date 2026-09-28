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


def replication():
    """Per-seed eval sys-SRCC of every replication system, with the paper's number as a line."""
    rows = {r["system"]: r for r in json.load(open("results/rep_summary.json"))}
    order = [("sslmos_16k", "SSL-MOS, 16 kHz"), ("hrm_model1_cv", "Model 1, selected on train labels (5-fold)"),
             ("hrm_model1", "Model 1"), ("hrm_model2", "Model 2"), ("hrm_model3", "Model 3"),
             ("hrm_model1_lrssl2e-5", "Model 1, SSL lr 2e-5"), ("hrm_model2_lrssl2e-5", "Model 2, SSL lr 2e-5"),
             ("hrm_model3_lrssl2e-5", "Model 3, SSL lr 2e-5"),
             ("HighRateMOS ensemble (ours)", "HighRateMOS ensemble (Model 1 best fold + 2 + 3)")]
    order = [(k, lab) for k, lab in order if k in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(order) + 1.3), sharey=True)
    for ax, metric, paper in [(axes[0], "sys_SRCC", 0.955), (axes[1], "utt_LCC", 0.847)]:
        for i, (k, lab) in enumerate(order[::-1]):
            seeds = [r[f"eval_{metric}"] for r in rows[k]["per_seed"]]
            ens = rows[k]["eval_ens"][metric]
            col = ps.SERIES[1] if k.startswith("HighRateMOS") else ps.SERIES[0]
            ax.scatter(seeds, [i] * len(seeds), s=30, color=col, edgecolor=ps.SURFACE, zorder=3)
            ax.scatter([ens], [i], s=90, marker="|", color=ps.INK, linewidth=2, zorder=4)
        ax.axvline(paper, color=ps.SERIES[7], lw=1.2, ls="--")
        ax.text(paper, len(order) - 0.45, f" paper {paper}", color=ps.INK2, fontsize=8, va="bottom")
        ax.set_xlabel(f"eval {metric.replace('_', ' ')}")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([lab for _, lab in order[::-1]], fontsize=8.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ps.SERIES[0], label="one training seed"),
               plt.Line2D([], [], marker="|", ls="", color=ps.INK, markersize=10, mew=2, label="seeds averaged"),
               plt.Line2D([], [], color=ps.SERIES[7], ls="--", label="HighRateMOS paper (T17)")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7.5)
    fig.suptitle("HighRateMOS replication on eval: one dot per training seed", fontsize=11, fontweight="bold",
                 color=ps.INK)
    ps.save(fig, FIG / "replication_seeds.png")


KIND_COLOR = {"published": ps.AXIS, "replication": ps.SERIES[0], "utmos": ps.SERIES[2],
              "ours-train": ps.SERIES[4], "ours": ps.SERIES[1]}
KIND_LABEL = {"published": "AudioMOS 2025 Track 3 teams (published)", "replication": "HighRateMOS, our replication",
              "utmos": "faster-UTMOSv2, off the shelf", "ours-train": "ours, train labels only",
              "ours": "ours, train + dev labels"}


def final_systems():
    """(label, kind, metrics) for every system in the headline comparison."""
    rows = [(n.replace("T17 HighRateMOS", "T17 HighRateMOS (paper)"), "published", r.to_dict())
            for n, r in PUBLISHED.iterrows()]
    rep = {r["system"]: r for r in json.load(open("results/rep_summary.json"))}
    if "HighRateMOS ensemble (ours)" in rep:
        rows.append(("HighRateMOS, replicated (3 seeds)", "replication", rep["HighRateMOS ensemble (ours)"]["eval_ens"]))
    ut = Path("results/utmosv2_summary.json")
    if ut.exists():
        rows.append(("faster-UTMOSv2, zero-shot", "utmos", json.load(open(ut))[0]["eval_ens"]))
    fin = Path("results/final/summary.json")
    if fin.exists():
        f = json.load(open(fin))
        if "train-only" in f:
            rows.append(("ours, train labels only", "ours-train", f["train-only"]["eval"]))
        if "train+dev" in f:
            rows.append(("ours, train + dev labels", "ours", f["train+dev"]["eval"]))
    return rows


def final_grid():
    """All 8 official metrics on eval, one panel each. Every panel uses the same row order
    (sorted by the primary metric, sys SRCC), so a row is the same system across panels."""
    rows = sorted(final_systems(), key=lambda r: r[2]["sys_SRCC"])
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(2, 4, figsize=(15, 0.36 * len(rows) * 2 + 2.4), sharey=True)
    y = np.arange(len(rows))
    for ax, metric in zip(axes.flat, COLUMNS):
        lower = metric.endswith("MSE")
        vals = [r[2][metric] for r in rows]
        best = min(vals) if lower else max(vals)
        for i, (lab, kind, m) in enumerate(rows):
            ax.barh(i, m[metric], height=0.66, color=KIND_COLOR[kind], edgecolor=ps.SURFACE)
            ax.text(m[metric], i, f" {m[metric]:.3f}", va="center", fontsize=7,
                    color=ps.INK if m[metric] == best else ps.INK2, fontweight="bold" if m[metric] == best else None)
        ax.set_xlim(0 if lower else max(0, min(vals) - 0.08), max(vals) * (1.28 if lower else 1.04))
        ax.set_title(metric.replace("_", " ") + (" (lower is better)" if lower else ""), fontsize=9.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
    kinds = [k for k in KIND_COLOR if any(r[1] == k for r in rows)]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in kinds], [KIND_LABEL[k] for k in kinds],
               loc="lower center", ncol=len(kinds), fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("AudioMOS 2025 Track 3 eval: all eight official metrics (rows sorted by sys SRCC; best value in bold)",
                 fontsize=12, fontweight="bold", color=ps.INK)
    ps.save(fig, FIG / "final_all_metrics.png")


def final_bars():
    rows = final_systems()
    eval_bars([(lab, m["sys_SRCC"], None, {"ours-train": "ours", **{k: k for k in KIND_COLOR}}[kind]) for lab, kind, m in rows],
              "final_sys_srcc.png", "sys_SRCC", "Track 3 eval: system-level SRCC (primary metric)")


def final_scatter():
    """Predicted vs true condition MOS on eval: HighRateMOS replication vs ours (train + dev)."""
    clips = pd.read_csv("data/clips.csv")
    test = clips[clips.split == "test"].set_index("clip")
    panels = [("results/preds/rep__HighRateMOS_ensemble_ours__eval.csv", "HighRateMOS, replicated"),
              ("results/final/both/eval.csv", "ours, train + dev labels")]
    panels = [p for p in panels if Path(p[0]).exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.3), squeeze=False)
    for ax, (path, title) in zip(axes[0], panels):
        pr = pd.read_csv(path).set_index("clip")["pred"]
        t = test.loc[pr.index].assign(pred=pr)
        g = t.groupby("condition").agg(pred=("pred", "mean"), true=("mos_mix", "mean"), sr=("sr_tag", "first"))
        srcc = stats.spearmanr(g.true, g.pred)[0]
        lo, hi = 1.5, 4.5
        ax.plot([lo, hi], [lo, hi], color=ps.AXIS, lw=1, ls="--")
        for sr in ["16k", "24k", "48k"]:
            q = g[g.sr == sr]
            ax.scatter(q.true, q.pred, s=46, color=ps.SR_COLOR[sr], marker=ps.SR_MARKER[sr], edgecolor=ps.SURFACE,
                       linewidth=1.2, label=sr, zorder=3)
        for c, r in g.iterrows():
            ax.annotate(c, (r.true, r.pred), xytext=(4, -3), textcoords="offset points", fontsize=6.3, color=ps.INK2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("true condition MOS (eval)")
        ax.set_ylabel("predicted condition MOS")
        ax.set_title(f"{title}: sys SRCC {srcc:.3f}")
        ax.legend(loc="upper left")
    ps.save(fig, FIG / "final_conditions.png")


def serving():
    rows = json.load(open("results/serving_bench.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    c = [r["concurrency"] for r in rows]
    for ax, key, lab in [(axes[0], "rtf", "RTF (audio seconds per wall second)"), (axes[1], "p50_ms", "p50 latency (ms)"),
                         (axes[2], "batch_avg", "average GPU batch size")]:
        ax.plot(c, [r[key] for r in rows], marker="o", color=ps.SERIES[1], lw=2)
        for x, r in zip(c, rows):
            ax.annotate(f"{r[key]:.0f}" if key != "batch_avg" else f"{r[key]:.1f}", (x, r[key]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color=ps.INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(c)
        ax.set_xticklabels(c)
        ax.set_xlabel("concurrent requests")
        ax.set_title(lab, fontsize=10)
    fig.suptitle("Serving the final ensemble (dynamic batching, one GPU)", fontsize=11, fontweight="bold", color=ps.INK)
    ps.save(fig, FIG / "serving.png")


if __name__ == "__main__":
    {"probes": probes, "ablation": ablation, "replication": replication, "final_grid": final_grid,
     "final_bars": final_bars, "final_scatter": final_scatter, "serving": serving}[sys.argv[1]]()
