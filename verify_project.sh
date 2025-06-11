#!/bin/bash
# ---------- SBATCH pre-amble (no real commands here) ------------------------
#SBATCH --job-name=anura_Animal2Vec
#SBATCH --partition=A100-IML
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%j.out          # optional, nicer file name

# ---------- runtime ---------------------------------------------------------
# Current working directory when you call “sbatch verify_project.sh”
WORKDIR="$(pwd)"
CONDA_HOME="/netscratch/$USER/miniconda3"
ENV_NAME="a2v_condaenv"

srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_22.06-py3.sqsh \
  --container-workdir="$WORKDIR" \
  --container-mounts="$WORKDIR:$WORKDIR,/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro" \
  bash -lc "
    set -euo pipefail
    source ${CONDA_HOME}/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
    pip install --requirement requirements.txt --quiet
    python animal2vec_train.py model.w2v_path=../animal2vec_large_pretrained_MeerKAT_240507.pt --config-dir=./configs/anuraset/ --config-name finetune_5pct
  "
