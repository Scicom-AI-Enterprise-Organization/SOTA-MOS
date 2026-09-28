"""faster-UTMOSv2 on Track 3: zero-shot, and fine-tuned under our protocols.

  python -m sotamos.utmos zeroshot --out exp/utmosv2/zeroshot/s0 --fold 0
  python -m sotamos.utmos finetune --out exp/utmosv2/ft_both_test/s0_f0 --protocol cv --fold 0 --seed 0 \
      --labels both --domains test

UTMOSv2 resamples every clip to 16 kHz. Its data-domain input is a one-hot over 10 slots that
feeds only the final linear layer, so each slot adds a constant. Fine-tuning reuses the slots:
  --domains test     single-rate tests -> "sarulab", mixed test -> "bvcc"
  --domains test_sr  one slot per (test, sampling rate), which gives the 16 kHz model the rate as a bias
Fine-tuning follows the fusion_stage3 recipe: AdamW 5e-5, weight decay 1e-4, per-step cosine,
batch 8, mixup (alpha 0.4), loss 0.7 * pairwise-difference (margin 0.2, L1) + 0.2 * MSE.
Validation and test use UTMOSv2's own TTA inference (random crops, silence trimming).
Outputs follow sotamos.train (pred_val.csv, pred_test.csv, val_curve.csv, result.json, done).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sotamos.data import load_clips, sentence_folds
from sotamos.metrics import evaluate

WAV_DIR = "data/post_eval_distro/wav"
RATES = (16000, 24000, 48000)
SLOTS = {
    "default": None,  # zero-shot: one slot for everything (--domain)
    "test": {**{("single", sr): "sarulab" for sr in RATES}, **{("mix", sr): "bvcc" for sr in RATES}},
    "test_sr": {
        ("single", 16000): "blizzard2008", ("single", 24000): "blizzard2009", ("single", 48000): "blizzard2010-EH1",
        ("mix", 16000): "blizzard2010-EH2", ("mix", 24000): "blizzard2010-ES1", ("mix", 48000): "blizzard2010-ES3",
    },
}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def slot_of(domains, test_name, sr, default="sarulab"):
    return default if SLOTS[domains] is None else SLOTS[domains][(test_name, int(sr))]


def predict(model, df, domains, test_name, reps, seed, device, default_slot="sarulab"):
    """TTA prediction through UTMOSv2's own inference path, one call per domain slot."""
    out = {}
    slots = df.sr.map(lambda sr: slot_of(domains, test_name, sr, default_slot))
    for slot, g in df.groupby(slots):
        set_seed(seed)
        res = model.predict(input_dir=WAV_DIR, val_list=[c[:-4] for c in g["clip"]], predict_dataset=slot,
                            num_repetitions=reps, batch_size=16, num_workers=6, device=device, verbose=False,
                            remove_silent_section=True)
        for r in res:
            out[Path(r["file_path"]).name] = r["predicted_mos"]
    model.train()
    return np.array([out[c] for c in df["clip"]])


def split(protocol, fold, clips):
    pool = clips[clips.split == "pool"].reset_index(drop=True)
    test = clips[clips.split == "test"].reset_index(drop=True)
    if protocol == "dev":
        return pool, pool, test
    fold_of = sentence_folds(clips, 5, 0)
    f = pool.sentence.map(fold_of)
    return pool[f != fold].reset_index(drop=True), pool[f == fold].reset_index(drop=True), test


def write_outputs(out, val_df, p_val, test, p_test, extra):
    pd.DataFrame({"clip": test["clip"].values, "pred": p_test}).to_csv(out / "pred_test.csv", index=False)
    pd.DataFrame({"clip": val_df["clip"].values, "pred": p_val}).to_csv(out / "pred_val.csv", index=False)
    json.dump(extra, open(out / "result.json", "w"), indent=1)
    (out / "done").write_text("ok\n")


def zeroshot(args, out, device):
    import utmosv2
    from utmosv2.dataset._utils import get_dataset_map

    clips = load_clips()
    pool = clips[clips.split == "pool"].reset_index(drop=True)
    test = clips[clips.split == "test"].reset_index(drop=True)
    model = utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=args.fold, device=device).to(device)
    p_pool = predict(model, pool, "default", "mix", args.reps, 1234, device, args.domain)
    p_test = predict(model, test, "default", "mix", args.reps, 1234, device, args.domain)
    # the domain one-hot is the last 10 inputs of the final linear layer: each slot is a constant offset
    w = model._model.fc.weight[0, -10:].detach().float().cpu().numpy()
    offsets = {name: float(w[i]) for name, i in get_dataset_map(model._cfg).items()}
    m = evaluate(p_pool, pool.mos_mix.values, pool.condition.values)
    write_outputs(out, pool, p_pool, test, p_test, {"best_step": 0, "best": m, "offsets": offsets})
    print(f"[done] zero-shot fold {args.fold} dev {json.dumps({k: round(v, 3) for k, v in m.items()})}", flush=True)


def finetune(args, out, device):
    import utmosv2
    from utmosv2.dataset._schema import DatasetItem
    from utmosv2.loss import PairwizeDiffLoss
    from utmosv2.utils import get_dataset

    set_seed(args.seed)
    clips = load_clips()
    train_df, val_df, test = split(args.protocol, args.fold, clips)
    init_fold = args.fold if args.fold >= 0 else args.seed % 5
    model = utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=init_fold, device=device).to(device)
    cfg = model._cfg
    cfg.dataset.remove_silent_section = True  # same trimming as predict()

    items = []
    for r in train_df.itertuples():
        if args.labels in ("single", "both"):
            items.append(DatasetItem(Path(r.path), slot_of(args.domains, "single", r.sr), float(r.mos_single)))
        if args.labels in ("mix", "both"):
            items.append(DatasetItem(Path(r.path), slot_of(args.domains, "mix", r.sr), float(r.mos_mix)))
    pred_test_name = "single" if args.labels == "single" else "mix"
    val_col = "mos_mix"

    loader = torch.utils.data.DataLoader(
        get_dataset(cfg, items, "train"), batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=6, persistent_workers=True, pin_memory=True,
        worker_init_fn=lambda _: torch.set_num_threads(1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader), eta_min=1e-8)
    pair, mse = PairwizeDiffLoss(margin=0.2, norm="l1"), nn.MSELoss()

    def crit(o, y):
        return 0.7 * pair(o, y) + 0.2 * mse(o, y)

    print(f"[data] {len(items)} training items from {len(train_df)} clips, val {len(val_df)}, "
          f"{len(loader)} steps/epoch, init fold {init_fold}", flush=True)
    best, best_state, bad, rows = {"score": -1e9, "step": -1}, None, 0, []
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for x1, x2, d, y in loader:
            x1, x2, d, y = x1.to(device), x2.to(device), d.to(device), y.to(device).float()
            lmd = float(np.random.beta(0.4, 0.4))
            perm = torch.randperm(x1.shape[0], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(lmd * x1 + (1 - lmd) * x1[perm], lmd * x2 + (1 - lmd) * x2[perm],
                          lmd * d + (1 - lmd) * d[perm]).squeeze(1).float()
            loss = lmd * crit(o, y) + (1 - lmd) * crit(o, y[perm])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
            step += 1
        p = predict(model, val_df, args.domains, pred_test_name, args.val_reps, 1234, device)
        m = evaluate(p, val_df[val_col].values, val_df.condition.values)
        rows.append(pd.DataFrame({"step": step, "clip": val_df["clip"].values, "pred": p}))
        s = sum(-m[k] if k.endswith("MSE") else m[k] for k in args.select.split("+"))
        if s > best["score"]:
            best = {"score": s, "step": step, "epoch": epoch, **m}
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"[eval] epoch {epoch} step {step} loss {run / len(loader):.4f} val utt_LCC {m['utt_LCC']:.3f} "
              f"sys_SRCC {m['sys_SRCC']:.3f} | best epoch {best['epoch']} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    p_val = predict(model, val_df, args.domains, pred_test_name, args.reps, 1234, device)
    p_test = predict(model, test, args.domains, pred_test_name, args.reps, 1234, device)
    pd.concat(rows).to_csv(out / "val_curve.csv", index=False)
    write_outputs(out, val_df, p_val, test, p_test, {"best_step": best["step"], "best": best,
                                                     "elapsed": time.time() - t0})
    print(f"[done] best epoch {best['epoch']} {json.dumps({k: round(v, 4) for k, v in best.items()})}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zeroshot", "finetune"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="cv", choices=["dev", "cv"])
    ap.add_argument("--labels", default="both", choices=["single", "mix", "both"])
    ap.add_argument("--domains", default="test", choices=["test", "test_sr"])
    ap.add_argument("--domain", default="sarulab", help="zero-shot data-domain slot")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--reps", type=int, default=5, help="TTA repetitions for final predictions")
    ap.add_argument("--val-reps", type=int, default=3, help="TTA repetitions for per-epoch validation")
    ap.add_argument("--select", default="utt_LCC")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SOTAMOS_THREADS", "8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"system": "utmosv2", **vars(args)}, open(out / "config.yaml", "w"), sort_keys=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    (zeroshot if args.mode == "zeroshot" else finetune)(args, out, device)


if __name__ == "__main__":
    main()
