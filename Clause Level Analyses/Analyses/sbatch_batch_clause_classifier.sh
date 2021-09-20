#!/bin/bash 

# job name:
#SBATCH -J Marij_Leg_Clause_Classification

# email error reports
#SBATCH --mail-user=aryan_srivastava@brown.edu
#SBATCH --mail-type=ALL

# output file
#SBATCH -o ./logs/aryan/comparsing-%j.out
#SBATCH -e ./logs/aryan/comparsing-%j.err

# Request runtime, memory, cores 
#SBATCH --time=48:00:00
#SBATCH --mem=7000
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --array=1-32

deactivate
source /users/asriva11/data/asriva11/clause_classification/env/bin/activate

python --version
export PYTHONUNBUFFERED=FALSE
python batch_classifier.py --array $SLURM_ARRAY_TASK_ID