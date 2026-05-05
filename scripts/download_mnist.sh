#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/MNIST/raw}"
mkdir -p "$ROOT"
BASE="https://raw.githubusercontent.com/fgnt/mnist/master"
FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

for f in "${FILES[@]}"; do
  out="$ROOT/$f"
  raw="$ROOT/${f%.gz}"
  if [[ -f "$raw" ]]; then
    echo "[ok] $raw exists"
    continue
  fi
  if [[ ! -f "$out" ]]; then
    echo "[download] $f"
    curl -L --retry 5 --fail -o "$out" "$BASE/$f"
  fi
  echo "[unzip] $f"
  gzip -dkf "$out"
done

echo "MNIST is ready under $ROOT"
