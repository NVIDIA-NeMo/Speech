#!/bin/bash

#SBATCH -p batch -A coreai_dlalgo_llm -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --gpus-per-node 8 --ntasks-per-node=8 --dependency=singleton

export WANDB_RESUME=allow
export WANDB_NAME=train_vae

DIR=`pwd`

srun --signal=TERM@300 -l --container-image ${IMAGE} --container-mounts "/lustre:/lustre/,/home:/home" --no-container-mount-home --mpi=pmix bash -c "cd ${DIR} ; python -u nemo/collections/diffusion/vae/train_vae.py --yes $*"
