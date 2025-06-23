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
  bash -c "conda activate animal2vec && python animal2vec_train.py model.w2v_path=../animal2vec_large_pretrained_MeerKAT_240507.pt --config-dir=./configs/anuraset/ --config-name finetune_5pct"
