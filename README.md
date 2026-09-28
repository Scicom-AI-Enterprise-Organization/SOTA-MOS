# SOTA-MOS

MOS prediction for speech at 16, 24 and 48 kHz: [AudioMOS Challenge 2025](https://sites.google.com/view/voicemos-challenge/past-challenges/audiomos-challenge-2025) Track 3.

The goal is to replicate [HighRateMOS](https://arxiv.org/abs/2506.21951), the Track 3 winner (team T17), and beat it.
We also compare against the challenge baseline SSL-MOS (B03) and [faster-UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2).

**Data:** [Scicom-intl/HighRateMOS-VoiceMOS2025](https://huggingface.co/Scicom-intl/HighRateMOS-VoiceMOS2025), a mirror of the challenge archives (model-type repo).

## The task

There are 20 conditions. Each is one generation method at one sampling rate: 4 at 16 kHz, 8 at 24 kHz, 8 at 48 kHz.
The methods are natural speech, WORLD vocoder, a neural vocoder and TTS, plus an AudioSR super-resolved version of each (`*SR`).
Each clip has 10 ratings on a 1–5 scale.

| split | clips | sentences | listening test | listeners |
|---|---|---|---|---|
| train | 400 (120 at 16k, 240 at 24k, 40 at 48k) | 30 | three single-rate tests, one per sampling rate | 01–10 |
| dev | the same 400 clips | 30 | one mixed-rate test | 01–10 |
| eval | 400 new clips | 30 others | the mixed-rate test, second half | 11–20 |

Train labels come from tests where every clip had the same sampling rate. Dev and eval labels come from a test that mixed all three rates.
The primary metric is **system-level SRCC** over the 20 conditions.

## Target

Official Track 3 evaluation results (HighRateMOS paper, Table I).

| system | utt MSE | utt LCC | utt SRCC | utt KTAU | sys MSE | sys LCC | sys SRCC | sys KTAU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B03 baseline | 0.273 | 0.821 | 0.695 | 0.508 | 0.119 | 0.941 | 0.749 | 0.547 |
| T08 | 0.277 | 0.811 | 0.716 | 0.529 | **0.056** | 0.978 | 0.913 | 0.758 |
| T11 | 0.282 | 0.813 | 0.714 | 0.536 | 0.085 | 0.968 | 0.917 | 0.789 |
| T13 | 0.298 | 0.796 | 0.671 | 0.487 | 0.090 | 0.972 | 0.926 | 0.779 |
| T16 | 0.287 | 0.830 | 0.723 | **0.589** | 0.071 | 0.952 | 0.891 | 0.750 |
| T19 | **0.238** | 0.846 | 0.694 | 0.513 | 0.080 | 0.955 | 0.914 | 0.758 |
| **T17 HighRateMOS** | 0.303 | **0.847** | **0.742** | 0.556 | 0.116 | **0.982** | **0.955** | **0.842** |

## The data

`scripts/analyze_data.py` produces every number and figure in this section.

### The mixed test punishes 16 kHz

![Same clips, two tests](results/figures/protocol_shift.png)

**All four 16 kHz conditions fall below the diagonal.** On average they drop 0.35 MOS once listeners also hear 24 and 48 kHz audio.
The 24 kHz conditions move by −0.05 and the 48 kHz ones by −0.04. The same ten listeners rated both tests.

The train labels still rank the dev conditions at sys-SRCC 0.874. A model that perfectly reproduced the train labels would be capped there on dev.

### Eval listeners favour high rates even more

![Condition MOS per test](results/figures/condition_mos.png)

Eval has new sentences and a new panel of listeners. Compared with dev, the 16 kHz conditions are unchanged, 24 kHz gains 0.17 and 48 kHz gains 0.36.
48k_tts moves from 3.36 to 4.14. Each 48 kHz condition averages only 5 clips.

### Knowing each clip's condition gets sys-SRCC 0.904

![Dev predicts eval](results/figures/dev_vs_eval_conditions.png)

These reference predictors are given each eval clip's condition and output that condition's MOS from another test:

| reference predictor on eval | utt MSE | utt LCC | utt SRCC | utt KTAU | sys MSE | sys LCC | sys SRCC | sys KTAU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev condition MOS | 0.201 | 0.894 | 0.819 | 0.666 | 0.100 | 0.980 | 0.904 | 0.811 |
| train condition MOS | 0.222 | 0.856 | 0.702 | 0.520 | 0.101 | 0.945 | 0.748 | 0.575 |

**Condition identity alone beats every Track 3 team on all four utterance-level metrics.**
The train-label reference scores sys-SRCC 0.748, the same as the B03 baseline (0.749).

HighRateMOS reaches sys-SRCC 0.955, above the dev reference. For comparison, two random halves of the eval listeners agree with each other at sys-SRCC 0.95.
Gaps of ±0.02 in sys-SRCC are within listener noise. The utterance-level reliability of eval labels is 0.90 (split-half, Spearman-Brown corrected).

### The sampling rate sets the bandwidth

![Long-term average spectrum per condition](results/figures/bandwidth.png)

Every clip fills the band up to its Nyquist frequency. At 24 kHz, the AudioSR versions reach 12 kHz while the plain versions roll off at about 11.5 kHz.
A model that resamples everything to 16 kHz sees the same 0–8 kHz band for 16k_original, 24k_original and 48k_original. Eval rates those three at 3.56, 4.08 and 4.34.

## Rules

- **The model sees audio and the sampling rate only.** The sampling rate comes from the file header.
  Eval filenames name the condition (`16k_sys1_utt31`). They are used only to group clips for system-level metrics.
- **Eval labels are never used for selection.** Checkpoints, configs and ensemble members are chosen on dev labels, or on out-of-fold predictions from 5-fold CV grouped by sentence.
- **Two label settings.** *train-only* uses the train labels, as in the HighRateMOS paper.
  *train+dev* adds the dev labels, which the challenge released at the start of its evaluation phase. Every result states which one it uses.

## Systems

| name | what it is | input rates |
|---|---|---|
| SSL-MOS | wav2vec2-base, SHEET recipe (B03-style baseline) | everything resampled to 16 kHz |
| HighRateMOS Models 1–3 | wav2vec2-base fed native-rate samples, sampling-rate embedding, multi-scale CNN on the mel spectrogram, BLSTM, FC. Model 2 adds cross-attention and Model 3 adds MFCC | native 16 / 24 / 48 kHz |
| HighRateMOS ensemble | Model 1 (best of 5 folds) + Models 2 and 3, as in paper Section V-D | native |
| faster-UTMOSv2 | the pretrained VMC 2024 winner, off the shelf (zero-shot), for comparison | resampled to 16 kHz |
| ours | SSL on the waveform + multi-scale CNN on the native-rate mel spectrogram (common 0–24 kHz axis) + sampling-rate and listening-test embeddings, trained on train+dev labels | native 16 / 24 / 48 kHz |

Our model reads every clip at its own sampling rate. The mel branch places all rates on one 0–24 kHz axis, so a 48 kHz clip contributes its 8–24 kHz band and a 16 kHz clip shows its 8 kHz ceiling.

## Experiments

Launched 2026-09-28 on 8×H20, 4 runs per GPU. Every run writes its dev (or out-of-fold) and eval predictions. Selection only reads the dev side.

| group | system | runs | protocol | labels | status |
|---|---|---:|---|---|---|
| replication | SSL-MOS 16 kHz | 3 seeds | dev selection | train | running |
| replication | HighRateMOS Model 1 | 3 seeds | dev selection | train | running |
| replication | HighRateMOS Model 2 | 3 seeds | dev selection | train | running |
| replication | HighRateMOS Model 3 | 3 seeds | dev selection | train | running |
| replication | HighRateMOS Model 1, 5-fold (training phase) | 3 seeds × 5 folds | CV on train labels | train | running |
| replication | Models 1–3 with SSL lr 2e-5 | 3 × 3 seeds | dev selection | train | queued |
| comparison | faster-UTMOSv2, pretrained, zero-shot | 5 folds | none | none | running |
| ours | 17 variants: base; labels mix-only / train-only; native-rate SSL input; no mel; no SR embedding; pooled head; weighted layers; BLSTM; condition-balanced sampling; HighRateMOS loss; 6 other backbones | 17 × 5 folds | 5-fold CV by sentence | train+dev | queued |
| probes | ridge on frozen SSL layers: 9 backbones × 2 input modes (16 kHz, native) | 18 feature sets | 5-fold CV by sentence | train+dev | features extracted |

**126 training runs** in total: 36 replication, 5 UTMOSv2, 85 ours. Plus 18 frozen-feature probes.

## Results

Running. Tables and figures land here as runs finish.

## Setup

Training runs on a remote GPU box. [claude-ping](https://github.com/Scicom-AI-Enterprise-Organization/claude-ping) keeps one persistent SSH connection open, and `uv` manages the environment.

```bash
CP=../claude-ping/claude-ping
$CP up
$CP sync
$CP env-sync                                              # HF_TOKEN from .env
$CP exec 'cd /root/SOTA-MOS && uv sync'
$CP exec 'cd /root/SOTA-MOS && uv run python scripts/prepare_data.py'
$CP exec 'cd /root/SOTA-MOS && uv run python scripts/analyze_data.py'
```

Train one model, or run a whole job file across the GPUs:

```bash
uv run python -m sotamos.train --config configs/hrm_model1.yaml --seed 0 --out exp/rep/hrm_model1/s0
uv run python scripts/make_jobs.py phase1
uv run python scripts/run_queue.py jobs/phase1.txt --gpus 0,1,2,3,4,5,6,7 --per-gpu 3
uv run python scripts/report.py rep
```

## Layout

```
sotamos/     data, features, models, losses, metrics, training, reporting
scripts/     data preparation, analysis, probes, job files, queue, ensembling, plots
configs/     experiment configs
results/     metric tables, predictions and figures
```
