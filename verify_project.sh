srun \  
--job-name=anura_Animal2Vec \  
--partition=A100-IML \  
--cpus-per-task=32 \  
--mem=200G \  
--time=2-00:00:00 \  
--gpus=1 \  
--container-image=/enroot/nvcr.io_nvidia_pytorch_22.06-py3.sqsh \  
--container-workdir="`pwd`" \  
--container-mounts="`pwd`":"`pwd`",/netscratch/$USER/results:"$(pwd)"/results,/ds:/ds:ro \  
bash -c "pip install -r requirements.txt && python testing_cuda.py"