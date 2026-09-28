#!/usr/bin/env python3
"""Where does a batch go? Times mel features, the shared SSL encodes and the fold heads.

  python profile_engine.py ../results/final/both/system.json [batch_size]
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_worker as pw  # noqa: E402
from engine import Engine  # noqa: E402

from sotamos.data import collate  # noqa: E402


def main():
    system, bs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    clips = pd.read_csv(root / "data/clips.csv")
    test = clips[clips.split == "test"].sample(bs, random_state=0)
    items = [(torch.from_numpy(n), sr, torch.from_numpy(r)) for n, sr, r, _ in
             (pw.decode((root / p).read_bytes()) for p in test.path)]
    eng = Engine(system)
    for _ in range(2):
        eng.score(items)
    torch.cuda.synchronize()
    spent = {"mel": 0.0}
    orig = eng._spec

    def timed_spec(*a, **k):
        t = time.perf_counter()
        r = orig(*a, **k)
        torch.cuda.synchronize()
        spent["mel"] += time.perf_counter() - t
        return r

    eng._spec = timed_spec
    t0 = time.perf_counter()
    eng.score(items)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    enc = 0.0
    for _, folds in eng.ft:
        model, cfg, _ = folds[0]
        wav, lens = collate([n if cfg["input_mode"] == "native" else r for n, _, r in items], True)
        t = time.perf_counter()
        with torch.no_grad(), eng._amp():
            model.ssl.encode(wav.to(eng.device), lens.to(eng.device))
        torch.cuda.synchronize()
        enc += time.perf_counter() - t
    audio = sum(len(r) for _, _, r in items) / 16000
    print(f"batch {bs} ({audio:.1f} s audio): total {total * 1000:.0f} ms | mel {spent['mel'] * 1000:.0f} ms | "
          f"SSL encodes {enc * 1000:.0f} ms | heads + rest {(total - spent['mel'] - enc) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
