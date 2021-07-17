#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J Marij_Leg_Subreddits


# email error reports
#SBATCH --mail-user=babak_hemmatian@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH -o subreddits-%j.out
#SBATCH -e subreddits-%j.err
# %A is the job id (as you find it when searching for your running / finished jobs on the cluster)
# %a is the array id of your current array job


# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH -c 2
#SBATCH -N 1

machine='ccv'

eval "$(conda shell.bash hook)"
conda deactivate
conda activate marijuana_study

export PYTHONUNBUFFERED=FALSE
python subreddit_extraction.py --machine $machine
