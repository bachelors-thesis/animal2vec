python scripts/anuraset_manifest.py \
  ../anuraset \
  --audio-dir audio \
  --dest manifests/validation \
  --valid-percent 0.2 \
  --n-split 5 \
  --seed 1612 \
  --limit-fraction 0.05


srun \  --job-name=anura_Yamnet\  --partition=A100-IML \  --cpus-per-task=32 \  --mem=200G \  --time=2-00:00:00 \  --container-image=/enroot/nvcr.io_nvidia_tensorflow_23.10-tf2-py3.sqsh \  --container-workdir="`pwd`" \  --container-mounts="`pwd`":"`pwd`",/netscratch/$USER/results:"$(pwd)"/results,/ds:/ds:ro \  bash -c "pip install -r requirements.txt && python main_classifier.py  --model yamnet --dataset_path /ds/audio/Bioacoustics/UAnuraSet --output_file_path /netscratch/$USER/results/classifier_results"