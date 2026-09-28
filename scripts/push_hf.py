"""Stage an exported system as a self-contained folder and upload it to Hugging Face.

  uv run python scripts/push_hf.py results/final/both/system.json --repo Scicom-intl/HighRateMOS-VoiceMOS2025
  uv run python scripts/push_hf.py exp/serving_test/system.json --dry-run      # stage only

Layout under <path-in-repo>/ (default model/):
  system.json                          members, paths relative to this folder
  members/<member>.pt                  probe fold models (ridge / kernel ridge)
  checkpoints/<system>/<run>/model.pt  fine-tuned fold checkpoints
  README.md                            model card
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.metrics import COLUMNS, PUBLISHED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def repo_token():
    """HF_TOKEN from this repo's .env. It wins over any HF_TOKEN a login profile exports
    (the remote box's profile sets a different, read-only token)."""
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ["HF_TOKEN"]


def card(system, stage, note=""):
    rows = []
    for m in system["members"]:
        if m["type"] == "probe":
            rows.append(f"| {m['name']} | {m['backbone']} | {m['input_mode'].replace('resample16k', '16 kHz')} | "
                        f"layer {m['layer']} | {m['kind']} on frozen features, 5 folds | {m['weight']} |")
        else:
            c = m["cfg"]
            depth = f"first {c['max_layers']} layers" if c.get("max_layers") else "all layers"
            how = "frozen SSL + trained head" if c.get("freeze") == "all" else "fine-tuned end to end"
            inp = ("native" if c["input_mode"] == "native" else "16 kHz") + (" + native mel" if c.get("mel") else "")
            rows.append(f"| {m['name']} | {c['backbone']} | {inp} | {c.get('layer', 'last')} of {depth} | "
                        f"{how}, {len(m['runs'])} folds | {m['weight']} |")
    ev = system.get("eval") or {}
    oof = system.get("oof") or {}
    t17 = PUBLISHED.loc["T17 HighRateMOS"]
    oof_table = ""
    if oof and not ev:  # interim release: dev numbers only
        oof_table = ("| out-of-fold on dev (400 clips) | " + " | ".join(c.replace("_", " ") for c in COLUMNS)
                     + " |\n|---|" + "---:|" * len(COLUMNS) + "\n| this model | "
                     + " | ".join(f"{oof[c]:.3f}" for c in COLUMNS) + " |\n")
    res = ""
    if ev:  # final release
        res = ("| eval (AudioMOS 2025 Track 3) | " + " | ".join(c.replace("_", " ") for c in COLUMNS) + " |\n|---|"
               + "---:|" * len(COLUMNS) + "\n"
               + "| **this model** | " + " | ".join(f"**{ev[c]:.3f}**" for c in COLUMNS) + " |\n"
               + "| HighRateMOS (T17, challenge winner) | " + " | ".join(f"{t17[c]:.3f}" for c in COLUMNS) + " |\n")
    text = f"""---
license: mit
tags: [audio, speech, mos, speech-quality-assessment, audiomos]
---

# SOTA-MOS: speech MOS at 16, 24 and 48 kHz

Predicts the mean opinion score (1–5) of synthetic speech as rated in a mixed-sampling-rate listening test
([AudioMOS Challenge 2025](https://sites.google.com/view/voicemos-challenge/past-challenges/audiomos-challenge-2025) Track 3).
Code, training and evaluation: [Scicom-AI-Enterprise-Organization/SOTA-MOS](https://github.com/Scicom-AI-Enterprise-Organization/SOTA-MOS).

{("> " + note + chr(10)) if note else ""}
{res}{oof_table}
## Ensemble

| member | backbone | input | layer | head | weight |
|---|---|---|---|---|---:|
{chr(10).join(rows)}

Members were chosen by greedy forward selection on out-of-fold predictions of the dev set. Eval labels were never used for selection.

## Sampling rates

Clips at 16, 24 or 48 kHz are scored as they are. Any other rate is resampled to the nearest of the three:
44.1 kHz → 48 kHz, 28 kHz → 24 kHz, 22.05 kHz → 24 kHz, 8 kHz → 16 kHz, 96 kHz → 48 kHz.

## Use

```bash
git clone https://github.com/Scicom-AI-Enterprise-Organization/SOTA-MOS && cd SOTA-MOS && uv sync --extra serve
hf download Scicom-intl/HighRateMOS-VoiceMOS2025 --include "model/*" --local-dir .
uv run python -m sotamos.predict --system model/system.json clip_44k.wav clip_16k.wav
# or the dynamic-batching server
cd serving && SYSTEM=../model/system.json bash run_serve.sh
```
"""
    (stage / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--repo", default="Scicom-intl/HighRateMOS-VoiceMOS2025")
    ap.add_argument("--path-in-repo", default="model")
    ap.add_argument("--stage", default="exp/hf_stage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default="", help="a line under the title (e.g. interim release)")
    args = ap.parse_args()
    system = json.load(open(args.system))
    stage = ROOT / args.stage / args.path_in_repo
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "members").mkdir(parents=True)
    for m in system["members"]:
        if m["type"] == "probe":
            dst = Path("members") / Path(m["path"]).name
            shutil.copy(ROOT / m["path"], stage / dst)
            m["path"] = str(dst)
        else:
            runs = []
            for r in m["runs"]:
                dst = Path("checkpoints") / Path(r).parent.name / Path(r).name
                (stage / dst).mkdir(parents=True, exist_ok=True)
                ck = torch.load(ROOT / r / "model.pt", map_location="cpu", weights_only=False)
                m["cfg"] = {k: ck["cfg"].get(k) for k in ("backbone", "freeze", "max_layers", "layer", "input_mode", "mel")}
                if ck["cfg"].get("freeze") == "all":  # the SSL is the untouched pretrained model: reload it from the hub
                    ck["state_dict"] = {k: v for k, v in ck["state_dict"].items() if not k.startswith("ssl.model.")}
                    ck["ssl_from_pretrained"] = True
                torch.save(ck, stage / dst / "model.pt")
                runs.append(str(dst))
            m["runs"] = runs
    card(system, stage, args.note)
    for m in system["members"]:
        m.pop("cfg", None)
    json.dump(system, open(stage / "system.json", "w"), indent=1)
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 1e6
    print(f"staged {stage} ({size:.0f} MB)")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    HfApi(token=repo_token()).upload_folder(
        folder_path=str(stage), repo_id=args.repo, repo_type="model", path_in_repo=args.path_in_repo,
        commit_message="Add SOTA-MOS final ensemble (AudioMOS 2025 Track 3)")
    print(f"uploaded to https://huggingface.co/{args.repo}/tree/main/{args.path_in_repo}")


if __name__ == "__main__":
    main()
