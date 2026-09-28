"""Run training jobs across GPUs, several per GPU.

  uv run python scripts/run_queue.py jobs/phase1.txt --gpus 0,1,2,3,4,5,6,7 --per-gpu 2

Job file: one job per line, `OUT_DIR<TAB>ARGS`, where ARGS go to
`python -m sotamos.train --out OUT_DIR ARGS`. ARGS starting with `-m MODULE` run that module
instead (e.g. `-m sotamos.utmos finetune ...`). Lines starting with # are ignored.
A job whose OUT_DIR/done exists is skipped, so a crashed queue can simply be re-run.
"""

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=2)
    args = ap.parse_args()

    q = queue.Queue()
    n = 0
    for jf in args.jobs:
        for line in open(jf):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            out, rest = line.split("\t", 1)
            if (Path(out) / "done").exists():
                continue
            q.put((out, rest))
            n += 1
    print(f"[queue] {n} jobs pending", flush=True)
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}

    def worker(gpu):
        while True:
            try:
                out, rest = q.get_nowait()
            except queue.Empty:
                return
            Path(out).mkdir(parents=True, exist_ok=True)
            parts = shlex.split(rest)
            module = "sotamos.train"
            if parts[:1] == ["-m"]:
                module, parts = parts[1], parts[2:]
            cmd = [sys.executable, "-m", module, "--out", out] + parts
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONWARNINGS="ignore")
            t0 = time.time()
            with open(Path(out) / "train.log", "a") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            with lock:
                stats["ok" if rc == 0 else "fail"] += 1
                print(f"[queue] gpu{gpu} rc={rc} {time.time() - t0:.0f}s {out}  "
                      f"(ok {stats['ok']} fail {stats['fail']} left {q.qsize()})", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",") for _ in range(args.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[queue] finished: ok {stats['ok']} fail {stats['fail']}", flush=True)


if __name__ == "__main__":
    main()
