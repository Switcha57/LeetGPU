#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
CUDA_BINARY="$SCRIPT_DIR/sparse_matrix_vector_multiplication_cuda"

"$CUDA_HOME/bin/nvcc" -std=c++17 \
    "$SCRIPT_DIR/sparse_matrix_vector_multiplication.cu" "$SCRIPT_DIR/run_sparse_matrix_vector_multiplication.cu" \
    -lcublas \
    -o "$CUDA_BINARY"

echo "CUDA:"
"$CUDA_BINARY"
echo "Triton:"
uv run python "$SCRIPT_DIR/run_sparse_matrix_vector_multiplication.py"
