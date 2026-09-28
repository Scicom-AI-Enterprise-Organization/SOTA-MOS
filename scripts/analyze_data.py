"""Label and audio analysis of the Track 3 data. Writes figures to results/figures/ and
a summary to results/data_analysis.json.

What it measures:
  - protocol shift: single-rate test MOS (train labels) vs mixed test MOS (dev labels), same clips
  - ceiling: dev condition MOS vs eval condition MOS (mixed test Part 1 vs Part 2)
  - label noise: split-half reliability of utterance MOS from the 10 ratings
  - bandwidth: long-term average spectrum per condition on a common 0-24 kHz axis
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos import plotstyle as ps  # noqa: E402
from sotamos.metrics import evaluate  # noqa: E402

ps.setup()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SYSTEM_ORDER = ["original", "nv", "tts", "world", "originalSR", "nvSR", "ttsSR", "worldSR"]


def sys_table(clips):
    pool, test = clips[clips.split == "pool"], clips[clips.split == "test"]
    t = pd.DataFrame(
        {
            "n_pool": pool.groupby("condition").size(),
            "n_test": test.groupby("condition").size(),
            "train_single": pool.groupby("condition").mos_single.mean(),
            "dev_mix": pool.groupby("condition").mos_mix.mean(),
            "test_mix": test.groupby("condition").mos_mix.mean(),
        }
    )
    t["sr_tag"] = [c.split("_")[0] for c in t.index]
    t["system"] = [c.split("_", 1)[1] for c in t.index]
    return t


def corr(a, b):
    return {
        "LCC": float(np.corrcoef(a, b)[0, 1]),
        "SRCC": float(stats.spearmanr(a, b)[0]),
        "KTAU": float(stats.kendalltau(a, b)[0]),
    }


def split_half(ratings, clips, test_name, part, n_rep=200, seed=0):
    """Spearman-Brown corrected split-half reliability of utterance and condition MOS."""
    rng = np.random.default_rng(seed)
    r = ratings[(ratings.test == test_name) & (ratings.part == part)]
    by_clip = r.groupby("clip")["rating"].apply(list)
    cond = clips.set_index("clip").condition
    utt_r, sys_r = [], []
    for _ in range(n_rep):
        a, b = {}, {}
        for clip, vals in by_clip.items():
            vals = np.array(vals)
            idx = rng.permutation(len(vals))
            h = len(vals) // 2
            a[clip], b[clip] = vals[idx[:h]].mean(), vals[idx[h:]].mean()
        a, b = pd.Series(a), pd.Series(b)
        rr = np.corrcoef(a, b)[0, 1]
        utt_r.append(2 * rr / (1 + rr))
        sa, sb = a.groupby(cond).mean(), b.groupby(cond).mean()
        rs = stats.spearmanr(sa, sb)[0]
        sys_r.append(rs)
    return float(np.mean(utt_r)), float(np.mean(sys_r))


def fig_protocol_shift(t):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lo, hi = 1.6, 4.3
    ax.plot([lo, hi], [lo, hi], color=ps.AXIS, lw=1, ls="--", zorder=1)
    for sr in ["16k", "24k", "48k"]:
        s = t[t.sr_tag == sr]
        ax.scatter(s.train_single, s.dev_mix, s=46, color=ps.SR_COLOR[sr], marker=ps.SR_MARKER[sr],
                   edgecolor=ps.SURFACE, linewidth=1.2, label=f"{sr} ({len(s)} conditions)", zorder=3)
    for c, r in t.iterrows():
        if r.system in ("original", "originalSR", "world", "tts") or abs(r.dev_mix - r.train_single) > 0.25:
            ax.annotate(c, (r.train_single, r.dev_mix), xytext=(4, -3), textcoords="offset points",
                        fontsize=7, color=ps.INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("MOS in its single-rate test (train labels)")
    ax.set_ylabel("MOS in the mixed-rate test (dev labels)")
    ax.set_title("Same 400 clips, two listening tests")
    ax.legend(loc="upper left")
    ps.save(fig, FIG / "protocol_shift.png")


def fig_dev_vs_test(t, c_dev_test):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lo, hi = 1.6, 4.3
    ax.plot([lo, hi], [lo, hi], color=ps.AXIS, lw=1, ls="--", zorder=1)
    for sr in ["16k", "24k", "48k"]:
        s = t[t.sr_tag == sr]
        ax.scatter(s.dev_mix, s.test_mix, s=46, color=ps.SR_COLOR[sr], marker=ps.SR_MARKER[sr],
                   edgecolor=ps.SURFACE, linewidth=1.2, label=sr, zorder=3)
    for c, r in t.iterrows():
        ax.annotate(c, (r.dev_mix, r.test_mix), xytext=(4, -3), textcoords="offset points", fontsize=6.5,
                    color=ps.INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("condition MOS, dev (mixed test Part 1, sentences A)")
    ax.set_ylabel("condition MOS, eval (mixed test Part 2, sentences B)")
    ax.set_title(f"Dev predicts eval: SRCC {c_dev_test['SRCC']:.3f}, KTAU {c_dev_test['KTAU']:.3f}")
    ax.legend(loc="upper left")
    ps.save(fig, FIG / "dev_vs_eval_conditions.png")


def fig_condition_bars(t):
    """Grouped horizontal bars: per condition, train (single-rate), dev (mix), eval (mix)."""
    order = []
    for sr in ["16k", "24k", "48k"]:
        for s in SYSTEM_ORDER:
            if f"{sr}_{s}" in t.index:
                order.append(f"{sr}_{s}")
    tt = t.loc[order]
    y = np.arange(len(tt))[::-1]
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    cols = [("train_single", "train: single-rate test", ps.SERIES[6]),
            ("dev_mix", "dev: mixed test Part 1", ps.SERIES[3]),
            ("test_mix", "eval: mixed test Part 2", ps.SERIES[4])]
    for i, (col, lab, colr) in enumerate(cols):
        ax.barh(y + (1 - i) * h, tt[col], height=h - 0.03, color=colr, label=lab, edgecolor=ps.SURFACE, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tt.index, fontsize=8)
    for lab in ax.get_yticklabels():
        lab.set_color(ps.SR_COLOR[lab.get_text().split("_")[0]])
    ax.set_xlim(1, 4.6)
    ax.set_xlabel("MOS")
    ax.set_title("Condition MOS in each listening test")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIG / "condition_mos.png")


def ltas(path, sr_out_bins, nfft_48k=2048):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    nfft = int(nfft_48k * sr / 48000)  # same Hz resolution for every rate
    f, p = signal.welch(x, fs=sr, nperseg=nfft, noverlap=nfft // 2)
    db = 10 * np.log10(p + 1e-12)
    return np.interp(sr_out_bins, f, db, right=np.nan)


def fig_bandwidth(clips):
    bins = np.linspace(0, 24000, 481)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    res = {}
    for ax, sr in zip(axes, ["16k", "24k", "48k"]):
        conds = [s for s in SYSTEM_ORDER if f"{sr}_{s}" in set(clips.condition)]
        for i, s in enumerate(conds):
            sub = clips[clips.condition == f"{sr}_{s}"]
            spec = np.nanmean(np.stack([ltas(p, bins) for p in sub.path]), 0)
            res[f"{sr}_{s}"] = spec
            ax.plot(bins / 1000, spec, lw=1.4, color=ps.SERIES[i], label=s)
        ax.set_title(f"{sr} conditions")
        ax.set_xlabel("frequency (kHz)")
        ax.set_xlim(0, 24)
    axes[0].set_ylabel("long-term average spectrum (dB)")
    axes[-1].legend(loc="upper right", ncol=2, fontsize=7.5)
    ps.save(fig, FIG / "bandwidth.png")
    # effective bandwidth: highest frequency within 50 dB of the 0.3-4 kHz band level
    bw = {}
    for c, spec in res.items():
        ref = np.nanmean(spec[(bins > 300) & (bins < 4000)])
        ok = np.where(spec > ref - 50)[0]
        bw[c] = float(bins[ok.max()]) if len(ok) else float("nan")
    return bw


def main():
    clips = pd.read_csv("data/clips.csv")
    ratings = pd.read_csv("data/ratings.csv", dtype={"listener": str})
    t = sys_table(clips)
    pool, test = clips[clips.split == "pool"], clips[clips.split == "test"]

    summary = {}
    summary["sentences"] = {
        "pool": int(pool.sentence.nunique()),
        "test": int(test.sentence.nunique()),
        "shared": int(len(set(pool.sentence) & set(test.sentence))),
    }
    summary["conditions"] = t.round(3).reset_index().to_dict(orient="records")
    summary["sys_train_single_vs_dev_mix"] = corr(t.train_single, t.dev_mix)
    summary["sys_dev_mix_vs_test_mix"] = corr(t.dev_mix, t.test_mix)
    summary["sys_train_single_vs_test_mix"] = corr(t.train_single, t.test_mix)
    summary["utt_single_vs_mix_pool"] = corr(pool.mos_single, pool.mos_mix)
    summary["shift_by_sr_dev_minus_train"] = (pool.groupby("sr_tag").mos_mix.mean()
                                              - pool.groupby("sr_tag").mos_single.mean()).round(3).to_dict()

    # oracle references on the eval set: predict each clip with its condition's MOS from another test
    for name, col in [("dev_condition_mean", "dev_mix"), ("train_condition_mean", "train_single")]:
        pred = test.condition.map(t[col])
        summary[f"oracle_{name}_on_eval"] = evaluate(pred, test.mos_mix, test.condition)

    ur, sr_ = split_half(ratings, clips, "mix", 2)
    summary["eval_split_half"] = {"utt_reliability_SB": ur, "sys_srcc_half_vs_half": sr_}
    ur, sr_ = split_half(ratings, clips, "mix", 1)
    summary["dev_split_half"] = {"utt_reliability_SB": ur, "sys_srcc_half_vs_half": sr_}
    n_lis = ratings.groupby(["test", "part"]).listener.nunique()
    summary["listeners_per_test"] = {f"{k[0]}_p{k[1]}": int(v) for k, v in n_lis.items()}
    summary["duration_s"] = clips.groupby("sr_tag").duration.describe()[["mean", "min", "max"]].round(2).to_dict()

    fig_protocol_shift(t)
    fig_dev_vs_test(t, summary["sys_dev_mix_vs_test_mix"])
    fig_condition_bars(t)
    summary["effective_bandwidth_hz"] = fig_bandwidth(clips)

    (OUT / "data_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    print(t.round(3).sort_values("test_mix").to_string())


if __name__ == "__main__":
    main()
