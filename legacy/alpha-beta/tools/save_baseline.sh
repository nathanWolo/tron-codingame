#!/usr/bin/env bash
# Freeze the current release build as the SPRT baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo build --release
mkdir -p "$ROOT/bin"
SRC="$ROOT/target/release/tron"
if [[ ! -e "$SRC" && -e "${SRC}.exe" ]]; then
  SRC="${SRC}.exe"
fi
DEST="$ROOT/bin/tron-baseline"
if [[ "$SRC" == *.exe ]]; then
  DEST="${DEST}.exe"
fi
cp "$SRC" "$DEST"
echo "Saved baseline -> $DEST"
ls -l "$DEST"
