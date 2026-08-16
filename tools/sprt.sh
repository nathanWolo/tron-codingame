#!/usr/bin/env bash
# Build the working tree as "dev" and SPRT it against bin/tron-baseline.
#
# Usage:
#   tools/save_baseline.sh          # once, when the current bot is the reference
#   # edit src/main.rs ...
#   tools/sprt.sh                   # default SPRT [0, 10] Elo
#   tools/sprt.sh --elo0 0 --elo1 5 --budget-ms 30
#   tools/sprt.sh --fixed --max-games 200
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${SPRT_BASELINE:-$ROOT/bin/tron-baseline}"
if [[ ! -x "$BASE" ]]; then
  echo "No baseline at $BASE" >&2
  echo "Run tools/save_baseline.sh first (or set SPRT_BASELINE=/path/to/bin)." >&2
  exit 2
fi

cargo build --release
DEV="${SPRT_DEV:-$ROOT/target/release/tron}"
mkdir -p "$ROOT/sprt-logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/sprt-logs/${STAMP}.jsonl"

exec python3 "$ROOT/tools/sprt.py" \
  --baseline "$BASE" \
  --dev "$DEV" \
  --log "$LOG" \
  "$@"
