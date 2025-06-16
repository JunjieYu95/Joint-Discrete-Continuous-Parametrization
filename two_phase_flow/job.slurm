#!/bin/bash
#SBATCH --job-name=model_calibration       # Job name
#SBATCH --output=output.log            # Output file name
#SBATCH --error=error.log              # Error file name
#SBATCH --partition=shanechi     # Partition (queue) name
#SBATCH --gres=gpu:v100:1                      # Number of nodes
#SBATCH --time=240:00:00                # Walltime (HH:MM:SS)
#SBATCH --ntasks=1                      # Number of MPI ranks
#SBATCH --mem-per-cpu=4G                       # Memory per node
#SBATCH --cpus-per-task=28

moduel purge
eval "$(conda shell.bash hook)"
module load gcc/11.3.0
conda activate pytorch_gpu

python model_calibration_cVAE_nonadjoint.py