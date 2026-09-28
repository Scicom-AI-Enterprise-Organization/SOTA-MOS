"""Write job files for scripts/run_queue.py.

  uv run python scripts/make_jobs.py phase1   # replication (train labels, dev selection)
  uv run python scripts/make_jobs.py phase2   # ours: CV ablations
"""

import sys
from pathlib import Path

SEEDS = [0, 1, 2]
FOLDS = range(5)


def dev_runs(name, config, seeds=SEEDS, extra=""):
    return [(f"exp/{name}/s{s}", f"--config {config} --seed {s} {extra}".strip()) for s in seeds]


def cv_runs(name, config, seeds=(0,), extra=""):
    return [(f"exp/{name}/s{s}_f{f}", f"--config {config} --seed {s} --fold {f} {extra}".strip())
            for s in seeds for f in FOLDS]


def phase1():
    jobs = []
    for m in ["sslmos_16k", "hrm_model1", "hrm_model2", "hrm_model3"]:
        jobs += dev_runs(f"rep/{m}", f"configs/{m}.yaml")
    # HighRateMOS training phase: 5-fold CV on train labels (held-out fold validated on train labels);
    # the fold model with the best dev score becomes ensemble member "Model 1 (best among 5-fold)"
    jobs += cv_runs("rep/hrm_model1_cv", "configs/hrm_model1.yaml", seeds=SEEDS,
                    extra="--set protocol=cv val_labels=single pred_pool=true")
    # same three models with a small SSL learning rate (paper's 1e-3 kept for the new layers)
    for m in ["hrm_model1", "hrm_model2", "hrm_model3"]:
        jobs += dev_runs(f"rep/{m}_lrssl2e-5", f"configs/{m}.yaml", extra="--set lr_ssl=2e-5")
    return jobs


def phase2():
    base = "configs/ours_base.yaml"
    variants = {
        "ours_base": "",
        "ours_labels_mix": "--set labels=mix test_emb=0",
        "ours_labels_single": "--set labels=single test_emb=0",
        "ours_native": "--set input_mode=native",
        "ours_nomel": "--set mel=false",
        "ours_nosr": "--set sr_emb=0",
        "ours_pool": "--set head=pool",
        "ours_weighted": "--set layer=weighted",
        "ours_blstm": "--set blstm=true",
        "ours_balance": "--set balance=condition",
        "ours_loss_hrm": "--set loss.l2=0 loss.l1=1.0 loss.contrastive=1.0 loss.lcc=1.0 loss_params.tau=0.0",
        "ours_wavlm_base": "--set backbone=microsoft/wavlm-base-plus",
        "ours_w2v2_large": "--set backbone=facebook/wav2vec2-large-lv60",
        "ours_wavlm_large": "--set backbone=microsoft/wavlm-large",
        "ours_d2v_large": "--set backbone=facebook/data2vec-audio-large",
        "ours_xlsr300m": "--set backbone=facebook/wav2vec2-xls-r-300m",
        "ours_hubert_large": "--set backbone=facebook/hubert-large-ll60k",
    }
    jobs = []
    for name, extra in variants.items():
        jobs += cv_runs(f"cv/{name}", base, extra=extra)
    return jobs


if __name__ == "__main__":
    phase = sys.argv[1]
    jobs = {"phase1": phase1, "phase2": phase2}[phase]()
    Path("jobs").mkdir(exist_ok=True)
    with open(f"jobs/{phase}.txt", "w") as f:
        for out, args in jobs:
            f.write(f"{out}\t{args}\n")
    print(f"jobs/{phase}.txt: {len(jobs)} jobs")
