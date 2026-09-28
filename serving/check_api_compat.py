#!/usr/bin/env python3
"""Contract test: is the SOTA-MOS server a drop-in replacement for faster-UTMOSv2's server?

  OURS=http://127.0.0.1:8765 THEIRS=http://127.0.0.1:8766 python check_api_compat.py clip.wav

For every case, both servers must return the same status code. For JSON bodies, every key the
faster-UTMOSv2 server returns must also come back from SOTA-MOS, with the same type and in the same
relative order. SOTA-MOS may add keys after them.
"""

import json
import os
import sys

import httpx

OURS = os.environ.get("OURS", "http://127.0.0.1:8765")
THEIRS = os.environ.get("THEIRS", "http://127.0.0.1:8766")


def call(base, method, path, **kw):
    r = httpx.request(method, base + path, timeout=300, **kw)
    ctype = r.headers.get("content-type", "")
    body = r.json() if "application/json" in ctype else None
    return r.status_code, ctype.split(";")[0], body


def compare(name, a, b):
    (sa, ca, ja), (sb, cb, jb) = a, b
    problems = []
    if sa != sb:
        problems.append(f"status {sa} vs {sb}")
    if ca != cb:
        problems.append(f"content-type {ca} vs {cb}")
    if isinstance(jb, dict) and sa == 200:
        missing = [k for k in jb if k not in (ja or {})]
        if missing:
            problems.append(f"missing keys {missing}")
        types = [k for k in jb if k in (ja or {}) and type(jb[k]) is not type(ja[k])
                 and not (isinstance(jb[k], (int, float)) and isinstance(ja[k], (int, float)) and k != "reps"
                          and k != "batch_size")]
        if types:
            problems.append("type differs: " + ", ".join(f"{k} {type(ja[k]).__name__} vs {type(jb[k]).__name__}" for k in types))
        shared = [k for k in ja if k in jb]
        if shared != [k for k in jb if k in ja]:
            problems.append("key order differs")
        extra = [k for k in ja if k not in jb]
    else:
        extra = []
    print(f"{'PASS' if not problems else 'FAIL'}  {name:34s} status {sa}/{sb}" + (f"  + extra keys {extra}" if extra else "")
          + ("" if not problems else "  <- " + "; ".join(problems)))
    return not problems


def main():
    wav = open(sys.argv[1], "rb").read()
    f = lambda: {"file": ("clip.wav", wav, "audio/wav")}  # noqa: E731
    cases = [
        ("POST /predict", "POST", "/predict", dict(files=f())),
        ("POST /predict?reps=3&dataset=bvcc", "POST", "/predict?reps=3&dataset=bvcc", dict(files=f())),
        ("POST /predict?reps=0 (invalid)", "POST", "/predict?reps=0", dict(files=f())),
        ("POST /predict?reps=17 (invalid)", "POST", "/predict?reps=17", dict(files=f())),
        ("POST /predict without file", "POST", "/predict", {}),
        ("GET /health", "GET", "/health", {}),
        ("GET /stats", "GET", "/stats", {}),
        ("POST /warmup", "POST", "/warmup", {}),
        ("GET / (upload form)", "GET", "/", {}),
    ]
    ok = True
    for name, method, path, kw in cases:
        a = call(OURS, method, path, **kw)
        b = call(THEIRS, method, path, **{k: (f() if k == "files" else v) for k, v in kw.items()})
        ok &= compare(name, a, b)
        if name == "POST /predict":
            print("      ours:  ", json.dumps(a[2]))
            print("      theirs:", json.dumps(b[2]))
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
