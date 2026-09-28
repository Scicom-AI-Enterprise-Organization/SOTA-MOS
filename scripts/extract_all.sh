#!/bin/bash
# Extract frozen features for every backbone x input mode, one job per GPU at a time.
#   bash scripts/extract_all.sh "0 1 2 3"
set -u
gpus=(${1:-0 1 2 3 4 5 6 7})
models=(facebook/wav2vec2-base facebook/wav2vec2-large-lv60 microsoft/wavlm-base-plus microsoft/wavlm-large
        facebook/data2vec-audio-large facebook/wav2vec2-xls-r-300m facebook/wav2vec2-xls-r-1b
        facebook/hubert-base-ls960 facebook/hubert-large-ll60k)
i=0
for mode in resample16k native; do
  for m in "${models[@]}"; do
    g=${gpus[$((i % ${#gpus[@]}))]}
    CUDA_VISIBLE_DEVICES=$g PYTHONWARNINGS=ignore uv run python scripts/extract_ssl_features.py --backbone "$m" --mode "$mode" &
    i=$((i + 1))
    if (( i % ${#gpus[@]} == 0 )); then wait; fi
  done
done
wait
echo EXTRACT_DONE
