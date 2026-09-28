"""Official AudioMOS/VoiceMOS metrics: MSE, LCC, SRCC, KTAU at utterance and system level.

For Track 3 a "system" is a condition (generation method x sampling rate). System-level
scores average the utterance predictions and labels within each condition first.
"""

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["MSE", "LCC", "SRCC", "KTAU"]
COLUMNS = [f"utt_{m}" for m in METRICS] + [f"sys_{m}" for m in METRICS]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return {
        "MSE": float(np.mean((y - p) ** 2)),
        "LCC": float(np.corrcoef(y, p)[0, 1]),
        "SRCC": float(stats.spearmanr(y, p)[0]),
        "KTAU": float(stats.kendalltau(y, p)[0]),
    }


def evaluate(pred: pd.Series, true: pd.Series, system: pd.Series) -> dict:
    """pred/true/system are aligned per utterance. Returns utt_* and sys_* metrics."""
    df = pd.DataFrame({"p": np.asarray(pred), "y": np.asarray(true), "s": np.asarray(system)})
    out = {f"utt_{k}": v for k, v in _metrics(df.y, df.p).items()}
    sys_df = df.groupby("s")[["p", "y"]].mean()
    out.update({f"sys_{k}": v for k, v in _metrics(sys_df.y, sys_df.p).items()})
    return out


def format_row(m: dict, digits: int = 3) -> str:
    return " ".join(f"{k}={m[k]:.{digits}f}" for k in COLUMNS if k in m)


# Official evaluation-set results, HighRateMOS paper (arXiv 2506.21951) Table I.
PUBLISHED = pd.DataFrame(
    [
        ["B03", 0.273, 0.821, 0.695, 0.508, 0.119, 0.941, 0.749, 0.547],
        ["T01", 0.303, 0.804, 0.643, 0.459, 0.104, 0.957, 0.866, 0.705],
        ["T08", 0.277, 0.811, 0.716, 0.529, 0.056, 0.978, 0.913, 0.758],
        ["T11", 0.282, 0.813, 0.714, 0.536, 0.085, 0.968, 0.917, 0.789],
        ["T13", 0.298, 0.796, 0.671, 0.487, 0.090, 0.972, 0.926, 0.779],
        ["T16", 0.287, 0.830, 0.723, 0.589, 0.071, 0.952, 0.891, 0.750],
        ["T19", 0.238, 0.846, 0.694, 0.513, 0.080, 0.955, 0.914, 0.758],
        ["T17 HighRateMOS", 0.303, 0.847, 0.742, 0.556, 0.116, 0.982, 0.955, 0.842],
    ],
    columns=["system"] + COLUMNS,
).set_index("system")
