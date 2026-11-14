#!/bin/bash
# Play the AlphaZero agent (a given network) against the alpha-beta engine over
# the CodinGame protocol, with the alpha-beta engine's SPRT referee.
#   bash bench_vs_baseline.sh <net.bin> [--players 4] [sprt.py options...]
cd "$(dirname "$0")"
export TRONZERO_NET="$(cygpath -w "$(realpath "$1")")"
shift
python legacy/alpha-beta/tools/sprt.py --baseline bin/tron-baseline.exe --dev "${DEV:-bin/tronzero-dev.exe}" "$@"
