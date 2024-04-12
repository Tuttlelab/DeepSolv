#!/bin/bash
#SBATCH --export=ALL
#SBATCH --output=crest.out
#SBATCH --job-name=crest
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time="01:00:00"
#SBATCH --ntasks=20 --nodes=1


module purge
module load nvidia/sdk/21.3
module load anaconda/python-3.9.7/2021.11
module load openmpi/intel-2020.4/1.4.5



/opt/software/scripts/job_prologue.sh

source ~/conda_initialise.sh
source activate Ross-Torch
python crest.py
source deactivate



/opt/software/scripts/job_epilogue.sh
