#!/bin/bash
# 123 
for seed in 11 231 321 312
do
    echo "Running experiment with seed $seed"
    python3 ../../../../../../baselines/sac/sac_memtasks.py \
        --env_id=RememberColor3-v0 \
        --exp-name=ppo-mlp-dense-remember-color-3-v0 \
        --capture-video \
        --no-save-model \
        --track \
        --num-steps=60 \
        --num-eval-steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --num-envs=32 \
        --eval-freq=100_000 \
        --utd=0.5 \
        --buffer_size=100 \
        --control-mode="pd_ee_delta_pos" \
        --camera_width=64 \
        --camera_height=64
done