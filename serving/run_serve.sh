#!/usr/bin/env bash
# Start (or restart) the SOTA-MOS server with sane defaults.
#   SYSTEM=../results/final/both/system.json MAX_BATCH=16 PP_WORKERS=16 bash run_serve.sh
set -uo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
# restart through a pidfile: `pkill -f uvicorn` would hit other people's servers on a shared box
PIDFILE="${PIDFILE:-/tmp/sota-mos-serve-$PORT.pid}"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill -9 "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 2
fi
echo $$ > "$PIDFILE"
export SYSTEM="${SYSTEM:-../results/final/both/system.json}"
export MAX_BATCH="${MAX_BATCH:-16}"
export MAX_WAIT_MS="${MAX_WAIT_MS:-10}"
export MAX_BATCH_SECONDS="${MAX_BATCH_SECONDS:-240}"
export PP_WORKERS="${PP_WORKERS:-8}"
export DTYPE="${DTYPE:-bf16}"
export WARMUP="${WARMUP:-1}"
# one uvicorn worker: a single GPU wants a single owner; scale with PP_WORKERS / MAX_BATCH
exec python -m uvicorn server:app --host "$HOST" --port "$PORT"
