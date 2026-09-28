"""Per-band mean/std of the 0-24 kHz log-mel (and MFCC) over time, in the probe feature format.

  uv run python scripts/spec_stats.py   ->  data/feats/specstats_mel.pt, data/feats/specstats_mfcc.pt
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import AudioCache, load_clips  # noqa: E402
from sotamos.features import SpecCache  # noqa: E402

torch.set_num_threads(8)
clips = load_clips()
cache = AudioCache(clips)
spec = SpecCache(cache, clips)
for which, store in [("mel", spec.mel), ("mfcc", spec.mfcc)]:
    F = torch.stack([torch.cat([store[c].mean(1), store[c].std(1)]) for c in clips["clip"]])[:, None, :]
    torch.save({"clips": list(clips["clip"]), "feats": F.half()}, f"data/feats/specstats_{which}.pt")
    print(which, tuple(F.shape))
