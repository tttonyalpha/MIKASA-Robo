#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 /home/jovyan/Kachaev_N/SimplerEnv/MIKASA-Robo/baselines/sac/sac_memtasks.py \
        --env_id=ShellGameTouch-v0 \
        --exp-name=sac-mlp-dense-shell-game-touch-v0 \
        --capture-video \
        --no-save-model \
        --no-track \
        --num-steps=90 \
        --num_eval_steps=90 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=7_500_000 \
        --num-envs=32 \
        --eval-freq=100_000 \
        --utd=0.5 \
        --buffer_size=100_000 \
        --control-mode="pd_ee_delta_pos" \
        --camera_width=64 \
        --camera_height=64
done
