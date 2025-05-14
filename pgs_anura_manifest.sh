srun \
  --job-name=anura_Animal2Vec \
  --partition=A100-IML \
  --gpus=1 \
  --cpus-per-task=32 \
  --mem=200G \
  --time=0-08:00:00 \
  --mail-type=ALL \
  --mail-user=renenowo98@gmail.com \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_22.06-py3.sqsh \
  --container-workdir="$(pwd)" \
  --container-mounts="$(pwd):$(pwd),/netscratch/$USER/results:$(pwd)/results,/ds:/ds:ro" \
  bash -c "python -m pip install --quiet --upgrade pip && \
          pip install --quiet -r requirements.txt && \
          python scripts/anuraset_manifest.py \
            /ds/audio/Bioacoustics/AnuraSet/anuraset \
            --audio-dir audio \
            --dest manifests/validation \
            --valid-percent 0.2 \
            --n-split 5 \
            --seed 1612 \
            --limit-fraction 0.05"
