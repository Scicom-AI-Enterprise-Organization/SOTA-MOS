"""Download the AudioMOS 2025 Track 3 archives from the HF mirror and build manifests.

Outputs under data/:
  clips.csv    one row per audio clip (800): path, sampling rate, condition, sentence,
               split (pool = train/dev clips, test = eval clips), MOS per listening test
  ratings.csv  one row per listener rating (long format): clip, test, listener, rating

Listening tests: "16k", "24k", "48k" are the single-rate tests (train labels), "mix" is the
mixed-rate test (dev labels for pool clips, eval labels for test clips).
"""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf

REPO = "Scicom-intl/HighRateMOS-VoiceMOS2025"
ARCHIVES = [
    "track3_obf.tar.gz",
    "audiomos2025-track3-eval-phase.zip",
    "track3_post_eval_distro.tar.gz",
    "amc2025_track3_val_answer.zip",
]
PREFIX = "audiomos-track3-"


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def download(raw_dir: Path):
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        if not (raw_dir / name).exists():
            hf_hub_download(REPO, name, repo_type="model", local_dir=raw_dir, token=os.environ.get("HF_TOKEN"))


def extract(raw_dir: Path, data_dir: Path):
    if not (data_dir / "track3_obf").exists():
        subprocess.run(["tar", "xzf", raw_dir / "track3_obf.tar.gz", "-C", data_dir], check=True)
    if not (data_dir / "post_eval_distro").exists():
        subprocess.run(["tar", "xzf", raw_dir / "track3_post_eval_distro.tar.gz", "-C", data_dir], check=True)
    if not (data_dir / "audiomos2025-track3-eval-phase").exists():
        subprocess.run(["unzip", "-oq", raw_dir / "audiomos2025-track3-eval-phase.zip", "-d", data_dir], check=True)
    if not (data_dir / "val_answer").exists():
        subprocess.run(["unzip", "-oq", raw_dir / "amc2025_track3_val_answer.zip", "-d", data_dir / "val_answer"], check=True)


def read_map(path: Path) -> dict:
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            k, v = line.split("|")
            out[k] = v
    return out


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def parse_clip(name: str, systems: list[str]):
    """audiomos-track3-24k_originalSR_1089_134686_000015_000002.wav -> (24k, originalSR, sentence)"""
    stem = name[len(PREFIX):].removesuffix(".wav")
    sr_tag, rest = stem.split("_", 1)
    # longest system name first so "nvSR" wins over "nv"
    for s in sorted(systems, key=len, reverse=True):
        if rest.startswith(s + "_"):
            return sr_tag, s, rest[len(s) + 1:]
    raise ValueError(f"cannot parse {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    load_env()
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    if not args.skip_download:
        download(raw_dir)
    extract(raw_dir, data_dir)

    post = data_dir / "post_eval_distro"
    sys_map = read_map(post / "sysMap.txt")  # system name -> sysN
    utt_map = read_map(post / "uttMap.txt")  # sentence id -> uttNN
    systems = list(sys_map)

    tests = {
        "16k": post / "train/utt_16k.csv",
        "24k": post / "train/utt_24k.csv",
        "48k": post / "train/utt_48k.csv",
        "mix_val": post / "val/utt_mix_val.csv",
        "mix_test": post / "test/utt_mix_test.csv",
    }
    ratings = []
    for test, path in tests.items():
        df = pd.read_csv(path, dtype={"lisID": str})
        df["test"] = "mix" if test.startswith("mix") else test
        df["part"] = 2 if test == "mix_test" else 1
        ratings.append(df.rename(columns={"uttID": "clip", "lisID": "listener"}))
    ratings = pd.concat(ratings, ignore_index=True)
    # listener ids are only unique inside one test/part; make them globally unique
    ratings["listener"] = ratings["test"] + "_p" + ratings["part"].astype(str) + "_" + ratings["listener"]

    rows = []
    for name in sorted(os.listdir(post / "wav")):
        if not name.endswith(".wav"):
            continue
        sr_tag, system, sentence = parse_clip(name, systems)
        utt = utt_map[sentence]
        info = sf.info(post / "wav" / name)
        rows.append(
            dict(
                clip=name,
                path=str(post / "wav" / name),
                sr=info.samplerate,
                sr_tag=sr_tag,
                system=system,
                condition=f"{sr_tag}_{system}",
                official_sys=f"{PREFIX}{sr_tag}_{sys_map[system]}",
                official_utt=f"{PREFIX}{sr_tag}_{sys_map[system]}_{utt}",
                sentence=sentence,
                utt=utt,
                corpus="hificaptain" if "Seikatsu" in sentence else "librittsr",
                speaker=sentence.split("-")[0] if "Seikatsu" in sentence else sentence.split("_")[0],
                duration=info.frames / info.samplerate,
                channels=info.channels,
                subtype=info.subtype,
            )
        )
    clips = pd.DataFrame(rows)
    assert (clips["sr_tag"].map({"16k": 16000, "24k": 24000, "48k": 48000}) == clips["sr"]).all(), "sr tag != header"

    # split from label membership: clips rated in Part 2 of the mixed test are the eval set
    test_clips = set(ratings.loc[ratings["part"] == 2, "clip"])
    pool_clips = set(ratings.loc[ratings["test"] != "mix", "clip"])
    assert not (test_clips & pool_clips)
    clips["split"] = clips["clip"].map(lambda c: "test" if c in test_clips else ("pool" if c in pool_clips else "none"))
    assert (clips["split"] != "none").all()

    agg = ratings.groupby(["clip", "test"])["rating"].agg(["mean", "count", "std"]).reset_index()
    for test_name, col in [("16k", "single"), ("24k", "single"), ("48k", "single"), ("mix", "mix")]:
        sub = agg[agg["test"] == test_name].set_index("clip")
        for stat in ["mean", "count", "std"]:
            key = f"mos_{col}" if stat == "mean" else f"{stat}_{col}"
            clips[key] = clips[key].fillna(clips["clip"].map(sub[stat])) if key in clips else clips["clip"].map(sub[stat])

    # sanity: obfuscated copies are byte-identical to the de-obfuscated audio
    obf_train = data_dir / "track3_obf/DATA/wav"
    obf_eval = data_dir / "audiomos2025-track3-eval-phase/DATA/wav"
    n_checked = 0
    for r in clips.itertuples():
        src = (obf_train if r.split == "pool" else obf_eval) / f"{r.official_utt}.wav"
        if src.exists():
            assert md5(src) == md5(Path(r.path)), f"audio mismatch {src}"
            n_checked += 1
    eval_list = [l.strip() for l in open(data_dir / "audiomos2025-track3-eval-phase/DATA/sets/eval_list.txt") if l.strip()]
    assert set(eval_list) == set(clips.loc[clips.split == "test", "official_utt"]), "eval list mismatch"

    # sanity: system means reproduce the official system-level answer files
    for path, split, col in [
        (post / "val/sys_mix_val.csv", "pool", "mos_mix"),
        (post / "test/sys_mix_test.csv", "test", "mos_mix"),
    ]:
        ref = pd.read_csv(path).set_index("sysID")["MOS"]
        ours = clips[clips.split == split].groupby("condition")[col].mean()
        ours.index = PREFIX + ours.index
        assert (ours.reindex(ref.index) - ref).abs().max() < 1e-6, f"system MOS mismatch vs {path}"

    clips.to_csv(data_dir / "clips.csv", index=False)
    ratings.to_csv(data_dir / "ratings.csv", index=False)
    print(f"clips: {len(clips)} ({(clips.split == 'pool').sum()} pool, {(clips.split == 'test').sum()} test)")
    print(f"ratings: {len(ratings)}  audio md5-verified against obfuscated copies: {n_checked}")
    print(clips.groupby(["split", "sr_tag"]).size().unstack())


if __name__ == "__main__":
    main()
