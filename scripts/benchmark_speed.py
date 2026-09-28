"""Inference speed on the 400 eval clips, one GPU, batch size 1 per clip for our system.

  uv run python scripts/benchmark_speed.py --system results/final/both/system.json

Times: faster-UTMOSv2 as we ran it (5 folds x 5 TTA), faster-UTMOSv2 fast mode (fold 0, 1 TTA),
and our final system. Reports clips per second and real-time factor (audio seconds / wall seconds).
"""

import argparse
import json
import time

import numpy as np
import torch

from sotamos.data import load_clips
from sotamos.predict import SystemScorer, load_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="results/speed.json")
    args = ap.parse_args()
    clips = load_clips()
    test = clips[clips.split == "test"]
    audio_sec = float(test.duration.sum())
    res = {}

    import utmosv2

    def run_utmos(folds, reps):
        t0 = time.time()
        for f in folds:
            m = utmosv2.create_model(pretrained=True, config="fusion_stage3", fold=f, device="cuda:0")
            m.predict(input_dir="data/post_eval_distro/wav", val_list=[c[:-4] for c in test["clip"]],
                      num_repetitions=reps, batch_size=16, device="cuda:0", verbose=False)
        torch.cuda.synchronize()
        return time.time() - t0

    run_utmos([0], 1)  # warm up cuDNN and the file cache
    for name, folds, reps in [("faster-UTMOSv2, 5 folds x 5 TTA", range(5), 5), ("faster-UTMOSv2, 1 fold x 1 TTA", [0], 1)]:
        dt = run_utmos(folds, reps)
        res[name] = {"seconds": dt, "clips_per_s": len(test) / dt, "rtf": audio_sec / dt}

    scorer = SystemScorer(args.system, "cuda")
    wavs = [load_audio(p) for p in test.path]
    for x, sr, r16 in wavs[:5]:
        scorer(x, sr, r16)
    torch.cuda.synchronize()
    t0 = time.time()
    preds = [scorer(x, sr, r16) for x, sr, r16 in wavs]
    torch.cuda.synchronize()
    dt = time.time() - t0
    res["ours (final system)"] = {"seconds": dt, "clips_per_s": len(test) / dt, "rtf": audio_sec / dt,
                                  "members": len(json.load(open(args.system))["members"])}
    assert np.isfinite(preds).all()
    json.dump(res, open(args.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:40s} {v['seconds']:7.1f} s  {v['clips_per_s']:6.1f} clips/s  RTF {v['rtf']:.0f}x")


if __name__ == "__main__":
    main()
