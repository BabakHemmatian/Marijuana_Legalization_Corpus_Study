#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J train_gru

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# email error reports
#SBATCH --mail-user=alexander_fengler@brown.edu 
#SBATCH --mail-type=ALL

# output file
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job

#SBATCH --output /users/afengler/batch_job_out/train_gru_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-3

source /users/afengler/.bashrc
conda deactivate
conda activate tf-gpu-py37
# module load python/3.7.4 cuda/10.0.130 cudnn/7.4 tensorflow/2.0.0_gpu_py37

machine='ccv'

python gru_language_model.py --machine $machine  --idx $SLURM_ARRAY_TASK_ID