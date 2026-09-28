"""Extract frozen SSL features: per clip, per hidden layer, mean and std over valid frames.

  uv run python scripts/extract_ssl_features.py --backbone facebook/wav2vec2-large-lv60 --mode resample16k

Writes data/feats/<name>_<mode>.pt: {"clips": [...], "feats": float16 [n_clips, n_layers, 2*D]}.
mode: resample16k (every clip at 16 kHz) or native (native-rate samples fed as if 16 kHz).
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import AudioCache, load_clips  # noqa: E402
from sotamos.model import SSLBackbone  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--mode", default="resample16k")
    ap.add_argument("--out-dir", default="data/feats")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SOTAMOS_THREADS", "8")))
    name = args.backbone.split("/")[-1]
    out = Path(args.out_dir) / f"{name}_{args.mode}.pt"
    if out.exists():
        print("exists", out)
        return
    clips = load_clips()
    cache = AudioCache(clips)
    ssl = SSLBackbone(args.backbone, layer="weighted").cuda().eval()
    feats = []
    for r in clips.itertuples():
        wav = cache.get(r.clip, args.mode)[None].cuda()
        lens = torch.tensor([wav.shape[1]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if ssl.do_normalize:
                wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)
            hs = ssl.model(wav, output_hidden_states=True).hidden_states
        h = torch.stack([x[0].float() for x in hs])  # [L, T, D]
        feats.append(torch.cat([h.mean(1), h.std(1)], -1).half().cpu())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clips": list(clips["clip"]), "feats": torch.stack(feats)}, out)
    print("wrote", out, tuple(feats[0].shape))


if __name__ == "__main__":
    main()
