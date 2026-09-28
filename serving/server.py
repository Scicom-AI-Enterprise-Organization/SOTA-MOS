"""FastAPI dynamic-batching MOS server for an exported SOTA-MOS system.

  SYSTEM=../results/final/both/system.json MAX_BATCH=16 PP_WORKERS=16 bash run_serve.sh

  bytes -> [process pool] decode; any rate -> nearest trained rate (16/24/48 kHz) + a 16 kHz copy
        -> [asyncio queue] dynamic batching: fire at MAX_BATCH clips, when the oldest clip has waited
                           MAX_WAIT_MS, or when the padded batch would exceed MAX_BATCH_SECONDS
        -> [GPU thread]    Engine.score(batch): each SSL backbone runs once per batch, padded with an
                           attention mask; probe and fine-tuned heads -> weighted MOS
        -> JSON {"mos", "reps", "gpu_ms", "total_ms", "batch_size", "input_sampling_rate", "sampling_rate", "duration_s"}

Clips vary in length. Each batch starts from the oldest waiting clip (so long clips never starve)
and adds the waiting clips closest to its length, while the padded size (clips x longest clip)
stays under MAX_BATCH_SECONDS of 16 kHz audio. Nothing CPU- or GPU-bound runs on the event loop.

Endpoints match faster-UTMOSv2's server, so it is a drop-in replacement:
  POST /predict (multipart 'file'; ?reps=N and ?dataset= accepted) -> {"mos", "reps", "gpu_ms", "total_ms",
  "batch_size"} plus "input_sampling_rate", "sampling_rate", "duration_s". GET /health . GET /stats .
  POST /warmup . GET /
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_worker  # noqa: E402
from engine import Engine  # noqa: E402

SYSTEM = os.environ.get("SYSTEM", "../results/final/both/system.json")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "16"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "10"))
MAX_BATCH_SECONDS = float(os.environ.get("MAX_BATCH_SECONDS", "240"))  # padded 16 kHz audio per batch
PP_WORKERS = int(os.environ.get("PP_WORKERS", "8"))
WARMUP = int(os.environ.get("WARMUP", "1"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[os.environ.get("DTYPE", "bf16")]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def log(m):
    print(m, flush=True)


log(f"[load] system={SYSTEM} device={DEV} dtype={DTYPE} max_batch={MAX_BATCH} wait={MAX_WAIT_MS}ms "
    f"budget={MAX_BATCH_SECONDS}s pp={PP_WORKERS}")
ENGINE = Engine(SYSTEM, DEV, DTYPE)
log(f"[load] {len(ENGINE.members)} members, backbones {sorted(ENGINE.ssl)}")

GPU = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
PP = (concurrent.futures.ProcessPoolExecutor(max_workers=PP_WORKERS, mp_context=mp.get_context("spawn"),
                                             initializer=preprocess_worker.init)
      if PP_WORKERS > 0 else concurrent.futures.ThreadPoolExecutor(max_workers=4))
Q: "asyncio.Queue" = None
STATS = {"requests": 0, "batches": 0, "batched_items": 0, "gpu_ms_sum": 0.0, "max_bs": 0,
         "audio_s": 0.0, "padded_s": 0.0}


def _score(items):
    t0 = time.perf_counter()
    clips = [(torch.from_numpy(it["native"]), it["sr"], torch.from_numpy(it["r16"])) for it in items]
    mos = ENGINE.score(clips)
    if DEV == "cuda":
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1000
    n = len(items)
    longest = max(len(it["r16"]) for it in items)
    STATS["batches"] += 1
    STATS["batched_items"] += n
    STATS["gpu_ms_sum"] += gpu_ms
    STATS["max_bs"] = max(STATS["max_bs"], n)
    STATS["audio_s"] += sum(len(it["r16"]) for it in items) / 16000
    STATS["padded_s"] += n * longest / 16000
    for it, m in zip(items, mos):
        it["loop"].call_soon_threadsafe(it["fut"].set_result, (float(m), gpu_ms, n))


def _cut(buf):
    """Next batch: the oldest waiting clip (so long clips never starve), then the waiting clips closest
    to its length, up to MAX_BATCH clips and MAX_BATCH_SECONDS of padded audio."""
    budget = MAX_BATCH_SECONDS * 16000
    anchor = min(range(len(buf)), key=lambda i: buf[i]["t"])
    first = buf.pop(anchor)
    buf.sort(key=lambda it: abs(len(it["r16"]) - len(first["r16"])))
    batch, rest, longest = [first], [], len(first["r16"])
    for it in buf:
        n = max(longest, len(it["r16"]))
        if len(batch) < MAX_BATCH and (len(batch) + 1) * n <= budget:
            batch.append(it)
            longest = n
        else:
            rest.append(it)
    return batch, rest


async def _batch_loop():
    loop = asyncio.get_event_loop()
    buf, t0 = [], None
    while True:
        while len(buf) < MAX_BATCH * 4:  # drain what is already queued
            try:
                buf.append(Q.get_nowait())
                t0 = t0 or time.monotonic()
            except asyncio.QueueEmpty:
                break
        full = len(buf) >= MAX_BATCH
        overdue = buf and (time.monotonic() - t0) * 1000 >= MAX_WAIT_MS
        if full or overdue:
            batch, buf = _cut(buf)
            t0 = time.monotonic() if buf else None
            await loop.run_in_executor(GPU, _score, batch)
            continue
        if buf:
            timeout = max(0.001, MAX_WAIT_MS / 1000 - (time.monotonic() - t0))
            try:
                buf.append(await asyncio.wait_for(Q.get(), timeout))
            except asyncio.TimeoutError:
                pass
        else:
            buf.append(await Q.get())
            t0 = time.monotonic()


def _warmup():
    rng = np.random.default_rng(0)
    for sr, sec, b in [(16000, 3, 1), (48000, 5, MAX_BATCH), (24000, 8, 4)]:
        x = torch.from_numpy((0.05 * rng.standard_normal(sr * sec)).astype("float32"))
        _, _, r16, _ = preprocess_worker.decode(_wav_bytes(x.numpy(), sr))
        ENGINE.score([(x, sr, torch.from_numpy(r16))] * b)
    log("[warmup] done")


def _wav_bytes(x, sr):
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


app = FastAPI(title="sota-mos-serve")


@app.on_event("startup")
async def _startup():
    global Q
    Q = asyncio.Queue()
    loop = asyncio.get_event_loop()
    if WARMUP:
        await loop.run_in_executor(GPU, _warmup)
    blob = _wav_bytes(np.zeros(16000, dtype="float32"), 16000)
    await asyncio.gather(*[loop.run_in_executor(PP, preprocess_worker.decode, blob) for _ in range(max(1, PP_WORKERS))])
    asyncio.create_task(_batch_loop())
    log("[startup] ready")


@app.get("/health")
async def health():
    # the keys faster-UTMOSv2's /health returns come first (same names, same types), then ours
    return {"ok": True, "config": "sota-mos", "fold": -1, "dtype": str(DTYPE), "device": DEV,
            "max_batch": MAX_BATCH, "max_wait_ms": MAX_WAIT_MS, "num_frames": 0, "spec_imgs": 0,
            "feature_device": DEV, "pp_workers": PP_WORKERS,
            "system": SYSTEM, "members": [m["name"] for m in ENGINE.members], "max_batch_seconds": MAX_BATCH_SECONDS}


@app.get("/stats")
async def stats():
    b = max(1, STATS["batches"])
    return {**STATS, "avg_batch": STATS["batched_items"] / b, "avg_gpu_ms": STATS["gpu_ms_sum"] / b,
            "padding_overhead": STATS["padded_s"] / max(1e-9, STATS["audio_s"]) - 1}


@app.post("/warmup")
async def warmup():
    await asyncio.get_event_loop().run_in_executor(GPU, _warmup)
    return await health()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    reps: int = Query(1, ge=1, le=16, description="accepted for faster-UTMOSv2 compatibility; SOTA-MOS is "
                                                  "deterministic, so every pass gives the same score"),
    dataset: str = Query(None, description="accepted for faster-UTMOSv2 compatibility and ignored"),
):
    raw = await file.read()
    loop = asyncio.get_event_loop()
    native, sr, r16, sr_in = await loop.run_in_executor(PP, preprocess_worker.decode, raw)
    t0 = time.perf_counter()
    fut = loop.create_future()
    STATS["requests"] += 1
    await Q.put({"native": native, "sr": sr, "r16": r16, "fut": fut, "loop": loop, "t": time.monotonic()})
    mos, gpu_ms, bs = await fut
    # faster-UTMOSv2's response keys first, same order and types; the rate fields are additions
    return JSONResponse({"mos": mos, "reps": reps, "gpu_ms": round(gpu_ms, 2),
                         "total_ms": round((time.perf_counter() - t0) * 1000, 2), "batch_size": bs,
                         "input_sampling_rate": sr_in, "sampling_rate": sr, "duration_s": round(len(native) / sr, 3)})


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SOTA-MOS</title>
<style>:root{color-scheme:light dark;--fg:#111;--muted:#666;--bg:#fafafa;--card:#fff;--line:#e3e3e3}
@media (prefers-color-scheme:dark){:root{--fg:#e8e8e8;--muted:#9aa0a6;--bg:#0f1114;--card:#181b1f;--line:#2a2e34}}
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
.wrap{max-width:560px;margin:0 auto;padding:32px 16px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:20px}h1{font-size:22px;margin:0 0 4px}p{color:var(--muted);margin:0 0 20px}
.mos{font-size:44px;font-weight:600;margin-top:16px}</style></head><body><div class="wrap">
<h1>SOTA-MOS</h1><p>Speech MOS at 16, 24 or 48 kHz (mixed-rate listening test scale, 1&ndash;5).</p>
<div class="card"><input type="file" id="f" accept="audio/*"><div class="mos" id="m"></div><div id="d"></div></div>
<script>document.getElementById('f').onchange=async e=>{const fd=new FormData();fd.append('file',e.target.files[0]);
const r=await fetch('/predict',{method:'POST',body:fd});const j=await r.json();
document.getElementById('m').textContent=j.mos.toFixed(2);
document.getElementById('d').textContent=j.input_sampling_rate+' Hz in, scored at '+j.sampling_rate+' Hz, '+j.duration_s+' s, '+j.total_ms+' ms';};</script>
</div></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
