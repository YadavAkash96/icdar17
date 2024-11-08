#!/bin/bash -l

#SBATCH --job-name=encode_per_page                        # Job name
#SBATCH --output=//logs_encode_per_page_job.txt  # Output log file
#SBATCH --error=//logs_encode_per_page_job.txt   # Error log file
#SBATCH --nodes=1                        		                      # Use a single node
#SBATCH --ntasks=1                                             # Number of tasks (1 for a single process)
#SBATCH --cpus-per-task=8                                    # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                                      # Request 1 A100 GPU
#SBATCH --partition=v100
#SBATCH --time=3:00:00                                         # Set a time limit of 6 hours
#SBATCH --export=NONE                                          # Do not export environment variables

unset SLURM_EXPORT_ENV                                         # Unset SLURM_EXPORT_ENV to avoid conflicts

# Load the required Python module
module add python

# Activate your virtual environment
source venv/bin/activate

# Change to the directory where your script is located (if needed)
cd icdar23/

# Run your Python script with all the required parameters
python encode_per_page.py > /logs_encode_page_job.txt 2>&1
