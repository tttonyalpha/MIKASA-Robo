#!/bin/bash
# 123 
for seed in 231 321 312
do
    echo "Running experiment with seed $seed"
#     python3 baselines/sac/sac_memtasks.py \
    python3 /home/jovyan/Kachaev_N/SimplerEnv/MIKASA-Robo/baselines/sac/sac_memtasks.py \
        --env_id=InterceptMedium-v0 \
        --exp-name=intercept-medium-v0 \
        --capture-video \
        --no-save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=270 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --eval-freq=100_000 \
        --num-envs=32 \
        --utd=0.5 \
        --buffer_size=300_000 \
        --control-mode="pd_ee_delta_pos" \
        --camera_width=64 \
        --camera_height=64
done