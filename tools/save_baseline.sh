#!/usr/bin/env bash
# Freeze the current release build as the SPRT baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo build --release
mkdir -p "$ROOT/bin"
cp "$ROOT/target/release/tron" "$ROOT/bin/tron-baseline"
echo "Saved baseline -> $ROOT/bin/tron-baseline"
ls -l "$ROOT/bin/tron-baseline"
