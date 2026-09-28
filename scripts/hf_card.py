"""Build the Hugging Face repo README (root) with result and serving figures, and upload it.

  uv run python scripts/hf_card.py --dry-run     # writes exp/hf_stage/root/README.md + assets/
  uv run python scripts/hf_card.py               # uploads README.md and assets/ to the repo root

Numbers come from results/final/summary.json, results/rep_summary.json, results/utmosv2_summary.json
and results/serving_bench*.json.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.metrics import COLUMNS, PUBLISHED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "Scicom-intl/HighRateMOS-VoiceMOS2025"
GITHUB = "https://github.com/Scicom-AI-Enterprise-Organization/SOTA-MOS"
FIGS = ["final_all_metrics.png", "final_conditions.png", "serving.png", "serving_latency.png", "protocol_shift.png",
        "replication_seeds.png", "ablation_oof.png"]


def row(name, m, bold=()):
    return "| " + name + " | " + " | ".join((f"**{m[c]:.3f}**" if c in bold else f"{m[c]:.3f}") for c in COLUMNS) + " |"


def build():
    fin = json.load(open(ROOT / "results/final/summary.json"))
    ours, ours_t = fin["train+dev"]["eval"], fin["train-only"]["eval"]
    rep = {r["system"]: r for r in json.load(open(ROOT / "results/rep_summary.json"))}["HighRateMOS ensemble (ours)"]["eval_ens"]
    ut = json.load(open(ROOT / "results/utmosv2_summary.json"))[0]["eval_ens"]
    t17 = PUBLISHED.loc["T17 HighRateMOS"].to_dict()
    best = {c: (PUBLISHED[c].min() if c.endswith("MSE") else PUBLISHED[c].max()) for c in COLUMNS}
    wins = [c for c in COLUMNS if (ours[c] < best[c] if c.endswith("MSE") else ours[c] > best[c])]
    sys_json = json.load(open(ROOT / "results/final/both/system.json"))
    bench = {k: {r["concurrency"]: r for r in json.load(open(ROOT / f"results/{f}"))}
             for k, f in [("ours", "serving_bench.json"), ("utmos", "serving_bench_utmosv2.json")]}
    head = "| eval (400 clips, 20 conditions) | " + " | ".join(c.replace("_", " ") for c in COLUMNS) + " |\n|---|" + "---:|" * len(COLUMNS)
    table = "\n".join([head, row("**SOTA-MOS (this model)**", ours, bold=wins),
                       row("SOTA-MOS, train labels only", ours_t),
                       row("HighRateMOS (T17, challenge winner, paper)", t17),
                       row("HighRateMOS, our replication (3 seeds)", rep),
                       row("faster-UTMOSv2, off the shelf", ut),
                       "| best published Track 3 system, per metric | " + " | ".join(f"{best[c]:.3f}" for c in COLUMNS) + " |"])
    import yaml

    mrows = []
    for m in sys_json["members"]:
        cfg = yaml.safe_load(open(ROOT / "exp/cv" / m["name"].split(":")[1] / "s0_f0/config.yaml"))
        mrows.append(f"| [{cfg['backbone']}](https://huggingface.co/{cfg['backbone']}) | first {cfg['max_layers']} | {len(m['runs'])} |")
    members = "\n".join(mrows)
    example = (ROOT / "results/example_predictions.csv").read_text().strip()
    srv = "\n".join(f"| {c} | {bench['ours'][c]['req_s']:.1f} | {bench['ours'][c]['rtf']:.0f}× | {bench['ours'][c]['p50_ms']:.0f} ms | "
                    f"{bench['utmos'][c]['req_s']:.1f} | {bench['utmos'][c]['rtf']:.0f}× | {bench['utmos'][c]['p50_ms']:.0f} ms |"
                    for c in sorted(bench["ours"]))
    return f"""---
license: mit
tags: [audio, speech, mos, speech-quality-assessment, audiomos, sampling-rate]
---

# SOTA-MOS: speech MOS at 16, 24 and 48 kHz

SOTA-MOS predicts the mean opinion score (1–5) of synthetic speech as listeners rate it in a test that mixes sampling rates.
It is built for [AudioMOS Challenge 2025](https://sites.google.com/view/voicemos-challenge/past-challenges/audiomos-challenge-2025) Track 3.
Code, training and every experiment: [{GITHUB.split('/', 3)[3]}]({GITHUB}).

**It beats every Track 3 system, including the winner HighRateMOS, on {len(wins)} of 8 official metrics:**
{', '.join(c.replace('_', ' ') for c in wins)}.

{table}

Eval labels were never used to build or select the model. Members were chosen by greedy forward selection on out-of-fold predictions of the dev set,
by a rule fixed before any of our systems was scored on eval. The eval set was scored once.

![All eight metrics](assets/final_all_metrics.png)

![Per-condition predictions](assets/final_conditions.png)

**SOTA-MOS keeps the order of the 20 conditions across sampling rates.** The replicated HighRateMOS puts every good condition near 3.6.
Ours spreads them out, and it puts 16 kHz audio below the same system at 24 or 48 kHz, as the listeners did.

## Use

```bash
git clone {GITHUB} && cd SOTA-MOS
uv sync --extra serve
hf download {REPO} --include "model/*" --local-dir .
uv run python -m sotamos.predict --system model/system.json clip_16k.wav clip_28k.wav clip_44k.wav clip_48k.wav clip_48k_world.wav
```

The same natural sentence at four rates, and a WORLD-vocoded copy at 48 kHz:

```
{example}
```

From Python:

```python
from sotamos.predict import SystemScorer, load_audio

scorer = SystemScorer("model/system.json", "cuda")
x, sr, x16 = load_audio("clip.wav")   # any rate: resampled to the nearest of 16 / 24 / 48 kHz
print(scorer(x, sr, x16))              # MOS on the mixed-rate listening-test scale
```

### Sampling rates

16, 24 and 48 kHz are scored as they are. Any other rate goes to the nearest of the three:

| input | scored at |
|---:|---:|
| 8 / 11.025 kHz | 16 kHz |
| 22.05 kHz | 24 kHz |
| 28 / 32 kHz | 24 kHz |
| 44.1 kHz | 48 kHz |
| 88.2 / 96 kHz | 48 kHz |

## Serve

A FastAPI server with dynamic batching. It follows [faster-UTMOSv2/serving](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2/tree/main/serving),
with batches built by clip length because SOTA-MOS scores the whole clip.

```bash
cd serving
SYSTEM=../model/system.json MAX_BATCH=16 PP_WORKERS=16 bash run_serve.sh
curl -X POST http://127.0.0.1:8000/predict -F file=@clip_44k.wav
# {{"mos": 3.79, "input_sampling_rate": 44100, "sampling_rate": 48000, "duration_s": 5.6, "batch_size": 1, ...}}
```

![Serving](assets/serving.png)

![Serving latency](assets/serving_latency.png)

One GPU, the 400 Track 3 eval clips (1,506 s of audio), same client for both servers:

| concurrency | SOTA-MOS req/s | RTF | p50 | faster-UTMOSv2 req/s | RTF | p50 |
|---:|---:|---:|---:|---:|---:|---:|
{srv}

**SOTA-MOS serves as fast as faster-UTMOSv2.** faster-UTMOSv2 scores one fold on a 3-second crop. SOTA-MOS scores the whole clip with 3 models × 5 folds.
Served scores match the offline scores within 0.007 MOS on average (Pearson r ≥ 0.9998 at batch 1, 8 and 32).

These numbers come from a batch former that always took the shortest waiting clips. Under 64 concurrent requests that starved long clips (p95 3.9 s).
The shipped server starts every batch from the oldest waiting clip instead. Its numbers will be re-measured on an idle GPU.

## Model

Three frozen self-supervised speech models, each followed by a small trained head, 5 cross-validation folds each:

| backbone (frozen) | layers kept | folds |
|---|---|---:|
{members}

Each head combines the backbone's early layers (weighted sum over the kept layers), a multi-scale CNN on the clip's native-rate mel spectrogram on a 0–24 kHz axis, and learned embeddings for the sampling rate and the listening test.
The mel branch is what lets a 16 kHz clip look different from a 48 kHz one. Training used the Track 3 train labels (single-rate tests) and the dev labels (mixed-rate test), with the listening test as an input.

![Where the data disagrees](assets/protocol_shift.png)

**The mixed-rate test drops 16 kHz audio by 0.35 MOS** against its single-rate test. Train labels alone cannot teach that.
The replication, the ablations and the frozen-feature probes are in the [GitHub README]({GITHUB}#results).

## Files

| path | contents |
|---|---|
| `model/` | the SOTA-MOS ensemble: `system.json`, fold heads, model card |
| `track3_obf.tar.gz` | Track 3 train/dev audio and train labels, as distributed by the organisers |
| `audiomos2025-track3-eval-phase.zip` | Track 3 eval audio |
| `track3_post_eval_distro.tar.gz` | post-evaluation release: all 800 clips, train / dev / eval ratings per listener |
| `amc2025_track3_val_answer.zip` | dev system-level answers |
| `assets/` | figures used on this page |

## References

- W. Ren et al., [HighRateMOS: Sampling-Rate Aware Modeling for Speech Quality Assessment](https://arxiv.org/abs/2506.21951), 2025.
- W.-C. Huang et al., [The AudioMOS Challenge 2025](https://arxiv.org/abs/2509.01336), 2025.
- K. Baba et al., [UTMOSv2](https://arxiv.org/abs/2409.09305), 2024; [faster-UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage = ROOT / "exp/hf_stage/root"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "assets").mkdir(parents=True)
    for f in FIGS:
        shutil.copy(ROOT / "results/figures" / f, stage / "assets" / f)
    (stage / "README.md").write_text(build())
    print(f"staged {stage}")
    if args.dry_run:
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    from push_hf import repo_token
    from huggingface_hub import HfApi

    api = HfApi(token=repo_token())
    api.upload_folder(folder_path=str(stage), repo_id=REPO, repo_type="model", path_in_repo=".",
                      allow_patterns=["README.md", "assets/*"],
                      commit_message="README: results, usage, sampling rates, serving, benchmarks")
    print(f"uploaded https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
