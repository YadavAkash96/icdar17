#!/bin/bash -l

#SBATCH --job-name=training_job                        # Job name
#SBATCH --output=/home/vault/iwi5/iwi5232h/resources/logs_densenet_tr1.txt  # Output log file
#SBATCH --error=/home/vault/iwi5/iwi5232h/resources/logs_densenet_tr1.txt   # Error log file
#SBATCH --nodes=1                                      # Use a single node
#SBATCH --ntasks=1                                     # Number of tasks (1 for a single process)
#SBATCH --cpus-per-task=8                          # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                              # Request 1 V100 GPU
#SBATCH --partition=v100	                            # Use the V100 partition
#SBATCH --time=23:00:00                                # Set a time limit of 12 hours
#SBATCH --export=NONE                                  # Do not export environment variables
	
unset SLURM_EXPORT_ENV                                         # Unset SLURM_EXPORT_ENV to avoid conflicts

# Load the required Python module
module add python

export WANDB_API_KEY=""

# Activate your virtual environment
source myenv/bin/activate

# Change to the directory where your script is located (if needed)
cd icdar23/

# Run your Python script with all the required parameters
WANDB_MODE=offline python main.py > /home/vault/iwi5/iwi5232h/resources/train_densenet_tr1.txt 2>&1

