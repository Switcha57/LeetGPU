#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
CUDA_BINARY="$SCRIPT_DIR/silu_cuda"

"$CUDA_HOME/bin/nvcc" -std=c++17 \
    "$SCRIPT_DIR/silu.cu" "$SCRIPT_DIR/run_silu.cu" \
    -o "$CUDA_BINARY"

echo "CUDA:"
"$CUDA_BINARY"
echo "Triton:"
python "$SCRIPT_DIR/run_silu.py"
