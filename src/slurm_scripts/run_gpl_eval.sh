#!/bin/sh
#SBATCH --job-name=eval_gpl        # the title of the job
#SBATCH --output=logs/eval_gpl.log     # file to which logs are saved
#SBATCH --time=12:00:00                # job time limit - full format is D-H:M:S
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gres=gpu:1                   # number of gpus (reduced)
#SBATCH --ntasks=1                     # number of tasks
#SBATCH --mem-per-gpu=40G              # memory allocation (reduced)
#SBATCH --partition=gpu                # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12              # number of allocated cores (reduced)

# Command to run the Python script inside the Singularity container
srun --nodes=1 --gres=gpu:1 --ntasks=1 --partition=gpu singularity exec --nv ./containers/container-transformers-pytorch-gpu.sif python3 notebooks/eval_gpl_models.py