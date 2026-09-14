#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
CUDA_BINARY="$SCRIPT_DIR/geglu_cuda"

"$CUDA_HOME/bin/nvcc" -std=c++17 \
    "$SCRIPT_DIR/geglu.cu" "$SCRIPT_DIR/run_geglu.cu" \
    -o "$CUDA_BINARY"

echo "CUDA:"
"$CUDA_BINARY"
echo "Triton:"
python "$SCRIPT_DIR/run_geglu.py"
