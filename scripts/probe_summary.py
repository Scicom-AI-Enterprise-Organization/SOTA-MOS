"""Table of the best layer per frozen-feature probe (out-of-fold metrics on dev labels).

  uv run python scripts/probe_summary.py   ->  results/probe_summary.csv
"""

import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("results/probe/*_best.json")):
    d = json.load(open(f))
    feats = d["feats"].split("/")[-1].removesuffix(".pt")
    backbone, _, mode = feats.rpartition("_")
    rows.append({"backbone": backbone or feats, "input": mode, "labels": d["labels"], "layer": d["best_layer"],
                 "alpha": d["alpha"], **{k.replace("oof_", ""): v for k, v in d.items() if k.startswith("oof_")}})
df = pd.DataFrame(rows).sort_values(["labels", "utt_LCC"], ascending=[True, False])
df.to_csv("results/probe_summary.csv", index=False)
cols = ["labels", "backbone", "input", "layer", "utt_LCC", "utt_SRCC", "utt_MSE", "sys_SRCC", "sys_KTAU", "sys_LCC"]
print(df[cols].round(3).to_string(index=False))
