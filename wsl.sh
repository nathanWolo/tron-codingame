#!/bin/bash
# Run a command in the workspace inside WSL with the JAX/ROCm venv on PATH.
source /etc/profile; source ~/.profile 2>/dev/null; cd "$(dirname "$0")"
export JAX_PLATFORMS=rocm,cpu
export PATH=/opt/slinky-venv/bin:$PATH
exec "$@"
