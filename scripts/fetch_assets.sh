#!/usr/bin/env bash
# Download the data and released checkpoints this repo uses (into data/ and artifacts/, both git-ignored).
# Usage: bash scripts/fetch_assets.sh [all|tinystories|sudoku|checkpoints]
set -euo pipefail
cd "$(dirname "$0")/.."
D=https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main
M=https://huggingface.co/hazyresearch/ncm-checkpoints/resolve/main
what=${1:-all}
get() { [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; echo "fetch $2"; curl -sSL --retry 5 -o "$2.tmp" "$1" && mv "$2.tmp" "$2"; }; }
if [[ $what == all || $what == tinystories ]]; then
  for f in bpe_vocab_8192.pickle bpe_merges_8192.pickle valid_bpe_8192.pickle train_bpe_8192_offsets.npy train_bpe_8192.bin; do
    get "$D/tinystories/$f" "data/tinystories/$f"; done
fi
if [[ $what == all || $what == sudoku ]]; then
  for l in easy medium hard; do for f in puzzles.pickle solutions.pickle; do get "$D/sudoku/$l/$f" "data/sudoku/$l/$f"; done; done
fi
if [[ $what == all || $what == checkpoints ]]; then
  for l in easy medium hard; do for f in config.yaml tokenizer.pt generator.pt; do get "$M/sudoku/$l/$f" "artifacts/sudoku/$l/$f"; done; done
  for f in config.yaml tokenizer.pt generator.pt bpe_vocab_8192.pickle bpe_merges_8192.pickle; do get "$M/tinystories/$f" "artifacts/tinystories/$f"; done
fi
echo done
