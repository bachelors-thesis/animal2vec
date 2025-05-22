python scripts/anuraset_manifest.py \
  ../anuraset/anuraset \
  --audio-dir wav8k \
  --dest manifests/validation \
  --valid-percent 0.2 \
  --n-split 5 \
  --seed 1612 \
  --limit-fraction 0.05