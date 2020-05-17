#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J Combined_NN

# priority
#SBATCH --account=ssloman

# email error reports
#SBATCH --mail-user=nathaniel_goodman@brown.edu 
#SBATCH --mail-type=ALL

# output file
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job


# Request runtime, memory, cores
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=0-3

module load anaconda/3-5.2.0
conda activate marijuana_study
##module load python/3.7.4 cuda/10.0.130 cudnn/7.4 tensorflow/2.0.0_gpu_py37

machine='ccv'

python Combined_NN_Model.py --machine $machine --idx $SLURM_ARRAY_TASK_ID