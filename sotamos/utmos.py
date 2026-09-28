"""faster-UTMOSv2, off the shelf, as a comparison system on Track 3.

  python -m sotamos.utmos --out exp/utmosv2/zeroshot/s0 --fold 0

Runs the pretrained fusion_stage3 model of one fold through UTMOSv2's own inference path
(5 TTA repetitions, silence trimming) on every pool and eval clip. UTMOSv2 resamples all audio to
16 kHz. Its data-domain one-hot feeds only the final linear layer, so the domain slot shifts every
score by a constant: it moves MSE and leaves the correlations unchanged. The offsets of all slots
are saved in result.json.
Outputs follow sotamos.train (pred_val.csv = pool clips, pred_test.csv = eval clips, done).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from sotamos.data import load_clips
from sotamos.metrics import evaluate

WAV_DIR = "data/post_eval_distro/wav"


def predict(model, df, slot, reps, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    res = model.predict(input_dir=WAV_DIR, val_list=[c[:-4] for c in df["clip"]], predict_dataset=slot,
                        num_repetitions=reps, batch_size=16, num_workers=6, device=device, verbose=False,
                        remove_silent_section=True)
    out = {Path(r["file_path"]).name: r["predicted_mos"] for r in res}
    return np.array([out[c] for c in df["clip"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="zeroshot", choices=["zeroshot"])  # older job files pass it
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0, help="pretrained UTMOSv2 fold (0-4)")
    ap.add_argument("--domain", default="sarulab", help="data-domain slot (UTMOSv2 default: sarulab)")
    ap.add_argument("--reps", type=int, default=5, help="TTA repetitions")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SOTAMOS_THREADS", "8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"system": "utmosv2", **vars(args)}, open(out / "config.yaml", "w"), sort_keys=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    import utmosv2
    from utmosv2.dataset._utils import get_dataset_map

    clips = load_clips()
    pool = clips[clips.split == "pool"].reset_index(drop=True)
    test = clips[clips.split == "test"].reset_index(drop=True)
    model = utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=args.fold, device=device).to(device)
    p_pool = predict(model, pool, args.domain, args.reps, 1234, device)
    p_test = predict(model, test, args.domain, args.reps, 1234, device)
    w = model._model.fc.weight[0, -10:].detach().float().cpu().numpy()
    offsets = {name: float(w[i]) for name, i in get_dataset_map(model._cfg).items()}
    m = evaluate(p_pool, pool.mos_mix.values, pool.condition.values)
    pd.DataFrame({"clip": pool["clip"].values, "pred": p_pool}).to_csv(out / "pred_val.csv", index=False)
    pd.DataFrame({"clip": test["clip"].values, "pred": p_test}).to_csv(out / "pred_test.csv", index=False)
    json.dump({"best_step": 0, "best": m, "offsets": offsets}, open(out / "result.json", "w"), indent=1)
    (out / "done").write_text("ok\n")
    print(f"[done] UTMOSv2 fold {args.fold} dev {json.dumps({k: round(v, 3) for k, v in m.items()})}", flush=True)


if __name__ == "__main__":
    main()
