#!/bin/bash
# filepath: /network/scratch/m/mousseat/CVI/mila/submit_slurm.sh

#SBATCH --job-name=cvi_experiments
#SBATCH --output=cvi_experiments_%A_%a.out
#SBATCH --error=cvi_experiments_%A_%a.err
#SBATCH --array=1-8  # One task per experiment
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1

cd /network/scratch/m/mousseat/CVI/mila

# Source environment
source .env

# List of functions to run
experiments=(
    run_taxi_vi
    run_taxi_pi
    run_taxi_cvi
    run_taxi_c51
    run_frozenlake_vi
    run_frozenlake_pi
    run_frozenlake_cvi
    run_frozenlake_c51
)

# Run the experiment for this array task
./run_experiments.sh ${experiments[$SLURM_ARRAY_TASK_ID-1]}