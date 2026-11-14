#!/bin/bash
# Build the engine and copy it to bin/. Set CARGO_TARGET_DIR to build somewhere else: under a
# long checkout path the default, engine/target, can exceed Windows' path-length limit.
set -e
cd "$(dirname "$0")/engine"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
mkdir -p ../bin
# The alpha-beta engine, for `tronzero vsbase`: its source minus the inner doc comment.
grep -v "^//!" ../legacy/alpha-beta/src/main.rs > src/baseline_src.rs
cargo build --release 2>&1 | grep -E "^(error|Finished)" -A 12 | head -60
# bin/tronzero.exe is the copy a training run executes (Windows locks it while it runs);
# builds land in tronzero-dev.exe. `bash build.sh promote` replaces the training copy.
cp "$CARGO_TARGET_DIR/release/tronzero.exe" ../bin/tronzero-dev.exe
if [ "$1" = promote ]; then until cp ../bin/tronzero-dev.exe ../bin/tronzero.exe 2>/dev/null; do sleep 1; done; echo promoted; fi
