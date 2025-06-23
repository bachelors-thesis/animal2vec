python scripts/anuraset_manifest.py \
  ../anuraset/anuraset \
  --audio-dir wav8k \
  --dest manifests_8k/anuraset_100pct/validation \
  --valid-percent 0.2 \
  --n-split 5 \
  --seed 1612 \
  --limit-fraction 1.0