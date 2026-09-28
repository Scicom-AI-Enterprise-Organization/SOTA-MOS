"""Train one model.

  python -m sotamos.train --config configs/x.yaml [--fold K] [--seed S] [--out DIR] [--set key=value ...]

Protocols (config `protocol`):
  dev   train on the 400 pool clips with train labels (labels=single); validate on the dev
        (mixed-test) labels of the same clips. This is the HighRateMOS evaluation-phase setup.
  cv    K folds over pool sentences. Train on the other folds (labels: single|mix|both) and
        validate on the held-out fold's dev labels.
  full  train on every pool clip for max_steps, no validation.

The run keeps the best state by the validation metric (early stopping with patience), then
predicts the eval clips once, as a mixed-test rating. Validation predictions are logged at
every evaluation. This script never computes eval-set metrics, so selection only sees
validation data.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from sotamos.data import SR_INDEX, TEST_INDEX, AudioCache, Batcher, collate, load_clips, make_examples, sentence_folds
from sotamos.features import SpecCache
from sotamos.losses import Criterion
from sotamos.metrics import evaluate
from sotamos.model import MOSModel

DEFAULTS = dict(
    backbone="facebook/wav2vec2-base", layer="last", freeze="none", input_mode="resample16k",
    d_ssl_proj=0, sr_emb=0, test_emb=0, mel=False, mfcc=False, xattn=False, blstm=False, head="frame",
    labels="single", protocol="dev", k_folds=5, fold_seed=0,
    loss={"l1_frame": 1.0}, loss_params={"tau": 0.5, "margin": 0.1},
    optimizer="sgd", lr=1e-3, lr_ssl=None, weight_decay=0.0, momentum=0.9, batch_size=16,
    max_steps=20000, eval_every=100, patience=20, select="sys_SRCC", warmup_steps=0, scheduler="none",
    grad_clip=1.0, balance="none", repetitive_pad=True,
    n_mels=80, n_mfcc=40, mel_win_ms=25.0, mel_hop_ms=10.0, mel_fmax=24000, mel_fmax_mode="fixed",
    val_labels="mix", pred_pool=False, save_ckpt=False, amp="bf16",
)


def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(open(path)) or {})
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cur = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = parse_value(v)
    if cfg["lr_ssl"] is None:
        cfg["lr_ssl"] = cfg["lr"]
    return cfg


def selection_score(m: dict, select: str) -> float:
    total = 0.0
    for key in select.split("+"):
        total += -m[key] if key.endswith("MSE") else m[key]
    return total


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_inputs(clip_names, cache, spec, cfg, sr_idx, test_idx, device):
    wavs = [cache.get(c, cfg["input_mode"]) for c in clip_names]
    wav, lens = collate(wavs, cfg["repetitive_pad"])
    b = {"wav": wav, "lens": lens, "sr_idx": torch.as_tensor(sr_idx), "test_idx": torch.as_tensor(test_idx)}
    if cfg["mel"]:
        b["mel"], b["mel_lens"] = spec.batch(clip_names, "mel")
    if cfg["mfcc"]:
        b["mfcc"], b["mfcc_lens"] = spec.batch(clip_names, "mfcc")
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def predict(model, df, cache, spec, cfg, device, test_name="mix", bs=16):
    """Length-sorted batches with the same tiled padding as training; scores pool valid frames only."""
    model.eval()
    dtype = torch.bfloat16 if cfg["amp"] == "bf16" else None
    names = list(df["clip"])
    srs = list(df["sr"])
    order = sorted(range(len(names)), key=lambda i: len(cache.get(names[i], cfg["input_mode"])))
    preds = np.zeros(len(names))
    for i in range(0, len(order), bs):
        idx = order[i: i + bs]
        b = build_inputs([names[j] for j in idx], cache, spec, cfg, [SR_INDEX[srs[j]] for j in idx],
                         [TEST_INDEX[test_name]] * len(idx), device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            preds[idx] = model(b)["score"].float().cpu().numpy()
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SOTAMOS_THREADS", "8")))  # the box is CPU-oversubscribed
    cfg = load_config(args.config, args.set)
    cfg["fold"], cfg["seed"] = args.fold, args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clips = load_clips()
    pool = clips[clips.split == "pool"].reset_index(drop=True)
    test = clips[clips.split == "test"].reset_index(drop=True)
    if cfg["protocol"] == "dev":
        assert cfg["labels"] == "single", "dev protocol validates on dev labels, so it cannot train on them"
        train_df, val_df = pool, pool
    elif cfg["protocol"] == "cv":
        fold_of = sentence_folds(clips, cfg["k_folds"], cfg["fold_seed"])
        f = pool.sentence.map(fold_of)
        train_df, val_df = pool[f != args.fold], pool[f == args.fold]
    elif cfg["protocol"] == "full":
        train_df, val_df = pool, pool.iloc[:0]
    else:
        raise ValueError(cfg["protocol"])

    t0 = time.time()
    cache = AudioCache(clips)
    spec = None
    if cfg["mel"] or cfg["mfcc"]:
        spec = SpecCache(cache, clips, cfg["n_mels"], cfg["n_mfcc"], cfg["mel_fmax"], cfg["mel_win_ms"],
                         cfg["mel_hop_ms"], cfg["mel_fmax_mode"])
    examples = make_examples(train_df, cfg["labels"])
    # validation labels: mixed test (dev labels) by default; "single" = held-out train labels
    val_col, val_test = ("mos_mix", "mix") if cfg["val_labels"] == "mix" else ("mos_single", "single")
    print(f"[data] train examples {len(examples)} from {len(train_df)} clips, val {len(val_df)}, "
          f"test {len(test)}  ({time.time() - t0:.1f}s)", flush=True)

    model = MOSModel(cfg).to(device)
    ssl_params = [p for n, p in model.named_parameters() if n.startswith("ssl.model") and p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("ssl.model") and p.requires_grad]
    groups = [{"params": ssl_params, "lr": cfg["lr_ssl"]}, {"params": other, "lr": cfg["lr"]}]
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(groups, momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(groups)
    else:
        raise ValueError(cfg["optimizer"])
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_factor(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        if cfg["scheduler"] == "cosine":
            prog = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return 1.0

    crit = Criterion(cfg["loss"], cfg["loss_params"])
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params {n_params / 1e6:.1f}M trainable {n_train / 1e6:.1f}M", flush=True)

    # sampling weights
    if cfg["balance"] == "none":
        sampler_examples = examples
    else:
        key = {"sr": lambda e: e.sr_idx, "condition": lambda e: train_df.set_index("clip").condition[e.clip]}[cfg["balance"]]
        groups_ = pd.Series([key(e) for e in examples])
        w = 1.0 / groups_.map(groups_.value_counts()).values
        rng = np.random.default_rng(args.seed + 1)
        idx = rng.choice(len(examples), size=len(examples) * 20, p=w / w.sum())
        sampler_examples = [examples[i] for i in idx]
    batcher = iter(Batcher(sampler_examples, cache, cfg["input_mode"], cfg["batch_size"], args.seed,
                           cfg["repetitive_pad"]))

    log = open(out / "log.jsonl", "a")
    val_rows = []
    best = {"score": -1e9, "step": -1}
    best_state = None
    bad = 0
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else None
    model.train()
    t0 = time.time()
    run_loss, run_terms, n_run = 0.0, {}, 0
    for step in range(1, cfg["max_steps"] + 1):
        raw = next(batcher)
        names = raw.pop("clips")
        b = {k: v.to(device) for k, v in raw.items()}
        if spec is not None:
            if cfg["mel"]:
                m, ml = spec.batch(names, "mel")
                b["mel"], b["mel_lens"] = m.to(device), ml.to(device)
            if cfg["mfcc"]:
                m, ml = spec.batch(names, "mfcc")
                b["mfcc"], b["mfcc_lens"] = m.to(device), ml.to(device)
        f = lr_factor(step - 1)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * f
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            o = model(b)
        loss, terms = crit(o, b["target"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        run_loss += float(loss)
        n_run += 1
        for k, v in terms.items():
            run_terms[k] = run_terms.get(k, 0.0) + v
        if not math.isfinite(float(loss)):
            print(f"[train] non-finite loss at step {step}", flush=True)
            break

        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            rec = {"step": step, "loss": run_loss / n_run, **{k: v / n_run for k, v in run_terms.items()},
                   "elapsed": round(time.time() - t0, 1)}
            run_loss, run_terms, n_run = 0.0, {}, 0
            if len(val_df):
                p = predict(model, val_df, cache, spec, cfg, device, val_test)
                m = evaluate(p, val_df[val_col].values, val_df.condition.values)
                rec.update({f"val_{k}": v for k, v in m.items()})
                val_rows.append(pd.DataFrame({"step": step, "clip": val_df["clip"].values, "pred": p}))
                s = selection_score(m, cfg["select"])
                if s > best["score"]:
                    best = {"score": s, "step": step, **m}
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                print(f"[eval] step {step} loss {rec['loss']:.4f} val utt_LCC {m['utt_LCC']:.3f} "
                      f"utt_SRCC {m['utt_SRCC']:.3f} sys_SRCC {m['sys_SRCC']:.3f} sys_LCC {m['sys_LCC']:.3f} "
                      f"| best {best['step']} ({time.time() - t0:.0f}s)", flush=True)
            else:
                print(f"[train] step {step} loss {rec['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if len(val_df) and cfg["patience"] and bad >= cfg["patience"]:
                print(f"[train] early stop at {step}, best step {best['step']}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_step = best["step"] if best_state is not None else step
    p_test = predict(model, test, cache, spec, cfg, device)
    pd.DataFrame({"clip": test["clip"].values, "pred": p_test}).to_csv(out / "pred_test.csv", index=False)
    if len(val_df):
        p_val = predict(model, val_df, cache, spec, cfg, device, val_test)
        pd.DataFrame({"clip": val_df["clip"].values, "pred": p_val}).to_csv(out / "pred_val.csv", index=False)
        pd.concat(val_rows).to_csv(out / "val_curve.csv", index=False)
    if cfg["pred_pool"]:
        p_pool = predict(model, pool, cache, spec, cfg, device)
        pd.DataFrame({"clip": pool["clip"].values, "pred": p_pool}).to_csv(out / "pred_pool.csv", index=False)
    if cfg["save_ckpt"]:
        torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                    "spec_stats": spec.stats if spec is not None else None}, out / "model.pt")
    json.dump({"best_step": final_step, "best": best, "elapsed": time.time() - t0}, open(out / "result.json", "w"),
              indent=1)
    (out / "done").write_text("ok\n")
    print(f"[done] best step {final_step} {json.dumps({k: round(v, 4) for k, v in best.items() if k != 'step'})}",
          flush=True)


if __name__ == "__main__":
    main()
