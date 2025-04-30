python scripts/anuraset_manifest.py \
  ../anuraset \
  --audio-dir audio \
  --dest manifests/validation \
  --valid-percent 0.2 \
  --n-split 5 \
  --seed 1612 \
  --limit-fraction 0.05
