"""Export a selected final ensemble as a runnable system.

  uv run python scripts/export_final.py both      # results/final/both/result.json -> system.json

Probe members are refit on the same folds and saved to exp/final/<group>/<member>.pt. The refit must
reproduce the saved out-of-fold and eval predictions, or the export stops. Fine-tuned members point
at their fold checkpoints (exp/cv/<system>/s*_f*/model.pt). Run with:
  uv run python -m sotamos.predict --system results/final/both/system.json audio.wav ...
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.data import load_clips  # noqa: E402
from sotamos.probes import fit_member, layer_features, oof_and_test  # noqa: E402

HF = {
    "wav2vec2-base": "facebook/wav2vec2-base", "wav2vec2-large-lv60": "facebook/wav2vec2-large-lv60",
    "wavlm-base-plus": "microsoft/wavlm-base-plus", "wavlm-large": "microsoft/wavlm-large",
    "data2vec-audio-large": "facebook/data2vec-audio-large", "wav2vec2-xls-r-300m": "facebook/wav2vec2-xls-r-300m",
    "wav2vec2-xls-r-1b": "facebook/wav2vec2-xls-r-1b", "hubert-base-ls960": "facebook/hubert-base-ls960",
    "hubert-large-ll60k": "facebook/hubert-large-ll60k",
}


def probe_spec(member):
    kind, name = member.split(":", 1)
    if kind == "ridge":
        best = json.load(open(f"results/probe/{name}_best.json"))
        stem = Path(best["feats"]).stem
        return stem, dict(kind="ridge", labels=best["labels"], alpha=best["alpha"], layer=best["best_layer"],
                          window=1), f"results/probe/{name}"
    best = json.load(open(f"results/probe_krr/{name}_best.json"))
    stem = Path(best["feats"]).stem
    return stem, dict(kind="krr", labels=best["labels"], alpha=best["alpha"], gamma_scale=best["gamma_scale"],
                      layer=best["layer"], window=best["layers"]), f"results/probe_krr/{name}"


def main():
    group = sys.argv[1]
    res = json.load(open(f"results/final/{group}/result.json"))
    clips = load_clips()
    out_dir = Path(f"exp/final/{group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    members = []
    for member, weight in res["members"].items():
        if member.startswith("ft:"):
            runs = sorted(str(p.parent) for p in Path(f"exp/cv/{member[3:]}").glob("s*_f*/model.pt"))
            assert len(runs) >= 5, f"{member}: fold checkpoints missing ({len(runs)})"
            members.append({"name": member, "weight": weight, "type": "finetuned", "runs": runs})
            continue
        stem, spec, prefix = probe_spec(member)
        d = torch.load(f"data/feats/{stem}.pt")
        order = clips.set_index("clip").loc[d["clips"]].reset_index()
        F = layer_features(d["feats"], spec["layer"], spec["window"])
        is_pool = (order.split == "pool").values
        pool, test = order[is_pool].reset_index(drop=True), order[~is_pool].reset_index(drop=True)
        states = fit_member(spec, F[is_pool], pool, clips)
        oof, tp = oof_and_test(states, spec["kind"], F[is_pool], F[~is_pool], pool, test, clips, spec["labels"])
        ref_o = pd.read_csv(f"{prefix}_oof.csv").set_index("clip")["pred"].reindex(pool["clip"]).values
        ref_t = pd.read_csv(f"{prefix}_test.csv").set_index("clip")["pred"].reindex(test["clip"]).values
        err = max(np.abs(oof - ref_o).max(), np.abs(tp - ref_t).max())
        assert err < 1e-6, f"{member}: refit differs from saved predictions by {err}"
        backbone, _, mode = stem.rpartition("_")
        path = out_dir / f"{member.replace(':', '__')}.pt"
        torch.save({"spec": spec, "states": states}, path)
        members.append({"name": member, "weight": weight, "type": "probe", "backbone": HF[backbone],
                        "input_mode": mode, "path": str(path), **{k: spec[k] for k in ("kind", "layer", "window")}})
        print(f"exported {member} (max refit error {err:.2e})")
    system = {"group": group, "members": members, "selection": "greedy forward selection on out-of-fold dev",
              "oof": res["oof"], "eval": res["eval"]}
    json.dump(system, open(f"results/final/{group}/system.json", "w"), indent=1)
    print(f"wrote results/final/{group}/system.json with {len(members)} members")


if __name__ == "__main__":
    main()
