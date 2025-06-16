#!/bin/bash
#SBATCH --job-name=model_calibration       # Job name
#SBATCH --output=output.log            # Output file name
#SBATCH --error=error.log              # Error file name
#SBATCH --partition=arcf     # Partition (queue) name
#SBATCH --mem-per-cpu=4G                       # Memory per node
#SBATCH --cpus-per-task=4
#SBATCH --time=240:00:00                # Walltime (HH:MM:SS)


moduel purge
eval "$(conda shell.bash hook)"
module load gcc/11.3.0
conda activate pytorch_gpu

python tester.py