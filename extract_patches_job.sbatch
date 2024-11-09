#!/bin/bash -l

#SBATCH --job-name=extract_patches_job                        # Job name
#SBATCH --output=/home/vault/iwi5/iwi5232h/resources/logs_extract_patches.txt  # Output log file
#SBATCH --error=/home/vault/iwi5/iwi5232h/resources/logs_extract_patches.txt   # Error log file
#SBATCH --nodes=1                        		                      # Use a single node
#SBATCH --ntasks=1                                             # Number of tasks (1 for a single process)
#SBATCH --cpus-per-task=8                                     # Number of CPU cores per task
#SBATCH --gres=gpu:1 
#SBATCH --partition=rtx3080
#SBATCH --time=8:00:00                                         # Set a time limit of 6 hours
#SBATCH --export=NONE                                          # Do not export environment variables

unset SLURM_EXPORT_ENV                                         # Unset SLURM_EXPORT_ENV to avoid conflicts

# Load the required Python module
module add python

# Activate your virtual environment
source myenv/bin/activate

# Change to the directory where your script is located (if needed)
cd icdar23/

# Run your Python script with all the required parameters
python helpers/extract_patches.py \
    --in_dir /home/vault/iwi5/iwi5232h/resources/icdar2017-train_binarized/ \
    --out_dir /home/vault/iwi5/iwi5232h/resources/icdar17_train \
    --num_of_clusters 5000 \
    --patches_per_page -1 \
    --sigma 2.5 \
    --black_pixel_thresh -1 \
    --white_pixel_thresh 0.95 \
    > /home/vault/iwi5/iwi5232h/resources/logs_extract_patches_log.txt 2>&1