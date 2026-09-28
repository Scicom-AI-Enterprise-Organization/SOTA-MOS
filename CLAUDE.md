# SOTA-MOS

MOS prediction for speech at 16, 24 and 48 kHz (AudioMOS 2025 Track 3).
The goal is to replicate HighRateMOS (arXiv 2506.21951, eval sys-SRCC 0.955 / KTAU 0.842) and beat it.

## Ground rules

- **Never select on eval labels.** Checkpoints, configs and ensemble members are picked on dev labels or on
  sentence-grouped CV out-of-fold predictions. `sotamos/train.py` never computes eval metrics. Only
  `scripts/report.py` scores eval, and only after a system is fixed.
- **Audio and sampling rate are the only inputs.** Eval filenames carry the condition (`16k_sys1_utt31`), but no
  model may use them as input. Condition names are used only for grouping in system-level metrics.
- Dev labels may be used for training: the challenge released them at the start of the evaluation phase.
  Always report which labels a system trained on (`labels: single | mix | both`).

## Data facts (verified by `scripts/prepare_data.py` and `scripts/analyze_data.py`)

- 800 clips. `pool` holds 400 clips: the train audio, which is also the dev audio. `test` holds the 400 eval clips.
  Pool and test use disjoint sentences.
- There are 20 conditions (system × rate): 4 at 16k, 8 at 24k and 8 at 48k. The 48k conditions have only
  5 clips each per split.
- Each pool clip has two labels. `mos_single` comes from its single-rate test (train labels) and `mos_mix` from
  the mixed-rate test Part 1 (dev labels). Test clips have `mos_mix` from mixed test Part 2 (eval labels).
- Listeners 01–10 rated every Part 1 test, and they are the same people across tests. Eval listeners 11–20 are
  a different panel.
- The mixed test drops 16k clips by 0.35 MOS relative to their single-rate test. Condition means from dev
  predict eval at sys-SRCC 0.904 only, and the 48k conditions are the noisy part.
- The HF mirror `Scicom-intl/HighRateMOS-VoiceMOS2025` is a **model**-type repo (`repo_type="model"`).

## Remote box

Training runs on `root@8.222.165.68:1023` (key `./scicom`) in `/root/SOTA-MOS`. Drive it through claude-ping
(`claude-ping.json` sits in the repo root and is gitignored):

```bash
CP=/Users/husein.z/Documents/claude-ping/claude-ping
$CP sync                                   # rsync repo -> remote (excludes data/, exp/, secrets)
$CP exec 'cd /root/SOTA-MOS && uv run python scripts/report.py rep'
$CP run --session q1 --log /root/SOTA-MOS/logs/q1.log "cd /root/SOTA-MOS && uv run python scripts/run_queue.py jobs/phase1.txt"
$CP session --session q1                   # alive? exit code? last lines
```

- Always use `uv` on the remote. For installs, the fast index is the Aliyun mirror:
  `UV_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/ UV_INSECURE_HOST=mirrors.cloud.aliyuncs.com`.
- The 8×H20 box is shared with other jobs. Ask the user before launching heavy GPU work if the GPUs are busy.
- The CPU is oversubscribed (load around 160 on 164 cores), so torch must run with few threads.
  `train.py` sets 8 threads (`SOTAMOS_THREADS`). Without this, setup takes 10× longer.
- `data/cache/` holds the decoded audio and spectral features. Delete it if `prepare_data.py` changes.

## Layout

| path | role |
|---|---|
| `scripts/prepare_data.py` | download archives, verify audio md5s and official answer files, write `data/clips.csv`, `data/ratings.csv` |
| `scripts/analyze_data.py` | label/bandwidth analysis → `results/data_analysis.json`, `results/figures/*.png` |
| `sotamos/data.py` | audio cache, sentence folds, examples, tiled-padding batches |
| `sotamos/features.py` | log-mel / MFCC on a common 0–24 kHz axis, cached |
| `sotamos/model.py` | `MOSModel`: SSL-MOS, HighRateMOS Models 1–3, and ours (test embedding, pooled head) |
| `sotamos/losses.py` | clipped L1/L2 (utt or frame), UTMOS contrastive, LCC, CCC |
| `sotamos/train.py` | one run; protocols `dev`, `cv`, `full`; writes `pred_val.csv`, `pred_test.csv`, `val_curve.csv` |
| `sotamos/report.py`, `scripts/report.py` | seed/fold ensembling, HighRateMOS ensemble, markdown tables |
| `scripts/make_jobs.py`, `scripts/run_queue.py` | job files (phase1–4, utmos) and a multi-GPU queue; jobs are claimed atomically (`.claim`), so several queues can share files |
| `scripts/probe.py`, `scripts/probe_krr.py` | ridge / kernel-ridge probes on frozen SSL layers (`data/feats/`) |
| `scripts/final.py` | the pre-registered final selection; `--no-eval` / `--interim` never touch eval labels |
| `scripts/export_final.py`, `scripts/push_hf.py`, `scripts/hf_card.py` | export a system (refit check), push `model/` and the root README to Hugging Face |
| `sotamos/predict.py` | CLI and `SystemScorer`: any sampling rate, nearest trained rate |
| `serving/` | FastAPI dynamic-batching server, API-compatible with faster-UTMOSv2; `check_api_compat.py`, `check_equivalence.py`, `benchmark.py` |

Run directories follow `exp/<group>/<system>/s{seed}` for dev/full runs and `exp/<group>/<system>/s{seed}_f{fold}` for CV.

## Final system

`results/final/both/system.json`: three frozen-SSL heads (data2vec-large 4 layers, XLS-R 300M 10, XLS-R 1B 14), 5 folds each.
Eval: utt LCC 0.888, utt SRCC 0.803, sys SRCC 0.968, sys KTAU 0.884 (HighRateMOS: 0.847 / 0.742 / 0.955 / 0.842).
Eval was scored once, by `scripts/final.py`. Do not re-select on eval.

## Gotchas

- `df.clip` is pandas' `DataFrame.clip` method. Always write `df["clip"]`.
- wav2vec2-base has a group-norm feature encoder with no attention mask. Batches are padded by tiling the clip,
  and pooling uses only the valid frames. Keep training and inference padding identical.
- HF SSL models get `apply_spec_augment=False` and `layerdrop=0`, as in mos-finetune-ssl (`mask=False`).
- System-level metrics are fragile, because the 48k conditions average just 5 clips. Look at utterance-level
  metrics and per-seed spread before believing a sys-SRCC gain.
- `claude-ping sync` runs rsync with `--delete`. Anything generated on the remote must be in `sync_excludes`
  (`results`, `logs`, `data`, `exp`, `uv.lock` are).
- Never `pgrep -f`/`pkill -f` with a pattern that also appears in your own command line: it kills your own shell.
  Use the bracket trick (`[s]otamos`) or exact PIDs. Killing a queue runner kills its children's parent chain too.
- `--set x=2e-5`: YAML 1.1 reads `2e-5` as a string; `parse_value` coerces it, but write `2.0e-5` in configs.
- The remote login profile exports its own read-only `HF_TOKEN`. `push_hf.py` reads the repo `.env` token explicitly.
- Concurrent first-time downloads race (UTMOSv2 checkpoints via wget). Pre-download once before fanning out jobs.
- The box is shared: after a run finishes, other jobs return to the GPUs. Benchmark only on an idle GPU, and run
  compared servers back to back.
- README style: terse sentences, a bold claim leading each finding, tables, and a PNG plot per result section.
  No "not X, it's Y" constructions, and no hardware details.
