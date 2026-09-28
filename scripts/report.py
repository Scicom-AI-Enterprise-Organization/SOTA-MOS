"""Print result tables for groups of systems and save them under results/.

  uv run python scripts/report.py rep          # replication (exp/rep/*) + HighRateMOS ensemble
  uv run python scripts/report.py cv           # CV ablations (exp/cv/*)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sotamos.metrics import PUBLISHED  # noqa: E402
from sotamos.report import fmt_table, highratemos_ensemble, np_default, summarize, to_jsonable  # noqa: E402


def main():
    group = sys.argv[1]
    out = Path("results")
    out.mkdir(exist_ok=True)
    systems = sorted(p for p in Path(f"exp/{group}").iterdir() if p.is_dir())
    sums = [summarize(p) for p in systems]
    if group == "rep":
        sums.append(highratemos_ensemble())
        sums.append(highratemos_ensemble(suffix="_lrssl2e-5") if Path("exp/rep/hrm_model1_cv_lrssl2e-5").exists()
                    else None)
    sums = [s for s in sums if s is not None]
    table = fmt_table(sums, PUBLISHED.loc[["B03", "T17 HighRateMOS"]])
    print(table)
    (out / f"{group}_table.md").write_text(table + "\n")
    json.dump([to_jsonable(s) for s in sums], open(out / f"{group}_summary.json", "w"), indent=1, default=np_default)
    pred_dir = out / "preds"
    pred_dir.mkdir(exist_ok=True)
    for s in sums:
        name = s["system"].replace(" ", "_").replace("(", "").replace(")", "")
        s["test_pred"].rename("pred").to_csv(pred_dir / f"{group}__{name}__eval.csv")
        s["sel_pred"].rename("pred").to_csv(pred_dir / f"{group}__{name}__{s['kind']}.csv")


if __name__ == "__main__":
    main()
