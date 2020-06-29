#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J Marij_Leg_Parse


# email error reports
#SBATCH --mail-user=babak_hemmatian@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH -J parsing
#SBATCH -o parsing-%j.out
#SBATCH -e parsing-%j.err
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job


# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --array=0-3

machine='ccv'

eval "$(conda shell.bash hook)"
conda deactivate
conda activate marijuana_study

export PYTHONUNBUFFERED=FALSE
python Parse.py --machine $machine --array $SLURM_ARRAY_TASK_ID
