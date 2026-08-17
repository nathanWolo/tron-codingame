#!/usr/bin/env bash
# Build the working-tree candidate and watch it play bin/tron-baseline.
#
# Usage:
#   tools/watch.sh
#   tools/watch.sh --budget-ms 20 --delay-ms 250
#   tools/watch.sh --players 3
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

exec python3 "$ROOT/tools/watch.py" \
  --baseline "$BASE" \
  --dev "$DEV" \
  "$@"
