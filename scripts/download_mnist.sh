#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-data/MNIST/raw}"
BASE="https://raw.githubusercontent.com/fgnt/mnist/master"
mkdir -p "$RAW_DIR"

files=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

for f in "${files[@]}"; do
  gz="$RAW_DIR/$f"
  raw="$RAW_DIR/${f%.gz}"
  if [[ -f "$raw" ]]; then
    echo "[mnist] found $raw"
    continue
  fi
  if [[ ! -f "$gz" ]]; then
    echo "[mnist] downloading $f"
    curl -L --retry 5 --fail -o "$gz" "$BASE/$f"
  fi
  echo "[mnist] decompressing $f"
  gzip -dkf "$gz"
done
