#!/usr/bin/env python3
"""Concurrency + throughput benchmark for the SOTA-MOS server.

Fires TOTAL requests over a set of clips, at most C in flight, for each C in CONC. Reports
throughput (req/s), RTF = audio-seconds scored / wall-second, client latency p50/p95/p99 and the
server-side batch size.

  URL=http://127.0.0.1:8000 CLIPS='../data/post_eval_distro/wav/*.wav' CONC=1,8,32,64 TOTAL=400 \
    python benchmark.py
"""

import asyncio
import glob
import json
import os
import statistics
import time

import httpx
import soundfile as sf

URL = os.environ.get("URL", "http://127.0.0.1:8000")
CONC = [int(x) for x in os.environ.get("CONC", "1,8,32,64").split(",")]
TOTAL = int(os.environ.get("TOTAL", "400"))
GLOB = os.environ.get("CLIPS", "../data/post_eval_distro/wav/*.wav")
OUT = os.environ.get("OUT", "")

FILES = sorted(f for pat in GLOB.split(":") for f in glob.glob(pat))
assert FILES, f"no clips under {GLOB}"
PAYLOADS = []
for f in FILES:
    info = sf.info(f)
    PAYLOADS.append((os.path.basename(f), open(f, "rb").read(), info.frames / info.samplerate))


async def _one(client, sem, payload, out):
    name, blob, dur = payload
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{URL}/predict", files={"file": (name, blob, "audio/wav")}, timeout=300)
            r.raise_for_status()
            out.append((time.perf_counter() - t0, dur, r.json()["batch_size"]))
        except Exception as e:
            out.append((time.perf_counter() - t0, dur, -1))
            print(f"  ! {name}: {e}")


async def _run(conc):
    sem = asyncio.Semaphore(conc)
    out = []
    payloads = [PAYLOADS[i % len(PAYLOADS)] for i in range(TOTAL)]
    audio_s = sum(p[2] for p in payloads)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 4)) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[_one(client, sem, p, out) for p in payloads])
        wall = time.perf_counter() - t0
    lat = sorted(o[0] * 1000 for o in out if o[2] >= 0)
    bss = [o[2] for o in out if o[2] > 0]
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]  # noqa: E731
    row = {"concurrency": conc, "ok": len(lat), "req_s": len(lat) / wall, "rtf": audio_s / wall,
           "p50_ms": statistics.median(lat), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
           "batch_avg": statistics.mean(bss), "batch_max": max(bss)}
    print(f"C={conc:<3} ok={row['ok']}/{TOTAL}  {row['req_s']:6.1f} req/s  RTF={row['rtf']:6.1f}x  "
          f"p50={row['p50_ms']:6.1f}  p95={row['p95_ms']:6.1f}  p99={row['p99_ms']:6.1f} ms  "
          f"batch avg={row['batch_avg']:.1f} max={row['batch_max']}", flush=True)
    return row


async def main():
    print(f"{len(FILES)} clips, {sum(p[2] for p in PAYLOADS):.1f} s audio; TOTAL={TOTAL} -> {URL}")
    rows = [await _run(c) for c in CONC]
    if OUT:
        json.dump(rows, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
