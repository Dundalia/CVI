#!/bin/bash
# filepath: /network/scratch/m/mousseat/CVI/mila/run_experiments.sh

# Source environment variables
source .env

# Function for each experiment (mirroring launch.json)
run_taxi_vi() {
    uv run python ../train.py config/taxi_vi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=taxi_vi_seed_0 logger.tags='["env_taxi","algo_vi","seed_0","metric_tderror"]'
}

run_taxi_pi() {
    uv run python ../train.py config/taxi_pi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=taxi_pi_seed_0 logger.tags='["env_taxi","algo_pi","seed_0","metric_tderror"]'
}

run_taxi_cvi() {
    uv run python ../train.py config/taxi_cvi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=taxi_cvi_seed_0 logger.tags='["env_taxi","algo_cvi","seed_0","metric_tderror"]'
}

run_taxi_c51() {
    uv run python ../train.py config/taxi_c51.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=taxi_c51_seed_0 logger.tags='["env_taxi","algo_c51","seed_0","metric_tderror"]'
}

run_frozenlake_vi() {
    uv run python ../train.py config/frozenlake_vi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=frozenlake_vi_seed_0 logger.tags='["env_frozenlake","algo_vi","seed_0","metric_tderror"]'
}

run_frozenlake_pi() {
    uv run python ../train.py config/frozenlake_pi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=frozenlake_pi_seed_0 logger.tags='["env_frozenlake","algo_pi","seed_0","metric_tderror"]'
}

run_frozenlake_cvi() {
    uv run python ../train.py config/frozenlake_cvi.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=frozenlake_cvi_seed_0 logger.tags='["env_frozenlake","algo_cvi","seed_0","metric_tderror"]'
}

run_frozenlake_c51() {
    uv run python ../train.py config/frozenlake_c51.yaml logger.do.online=true logger.project_name=CVI-RL logger.run_name=frozenlake_c51_seed_0 logger.tags='["env_frozenlake","algo_c51","seed_0","metric_tderror"]'
}

# Run all (for SLURM batch)
run_all() {
    run_taxi_vi &
    run_taxi_pi &
    run_taxi_cvi &
    run_taxi_c51 &
    run_frozenlake_vi &
    run_frozenlake_pi &
    run_frozenlake_cvi &
    run_frozenlake_c51 &
    wait
}

# Usage: ./run_experiments.sh <function_name>
# E.g., ./run_experiments.sh run_taxi_vi  # Dry run one
# Or ./run_experiments.sh run_all        # Run all in parallel locally
if [ $# -eq 0 ]; then
    echo "Usage: $0 <function_name> (e.g., run_taxi_vi or run_all)"
    exit 1
fi

"$@"