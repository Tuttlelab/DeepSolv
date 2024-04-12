#!/bin/bash
#SBATCH --export=ALL
#SBATCH --output=DeepSolv_Boltzmann_avo_crest_repeat_3.out
#SBATCH --job-name=DeepSolv_Boltzmann_avo_crest_repeat_3
#SBATCH --account=tuttle-rmss
#SBATCH --partition=gpu --gpus 1
#SBATCH --time="48:00:00"
#SBATCH --ntasks=10 --nodes=1


module purge
module load nvidia/sdk/21.3
module load anaconda/python-3.9.7/2021.11



/opt/software/scripts/job_prologue.sh

source ~/conda_initialise.sh
source activate Ross-Torch
python -u DeepSolv_Boltzmann.py
source deactivate



/opt/software/scripts/job_epilogue.sh
