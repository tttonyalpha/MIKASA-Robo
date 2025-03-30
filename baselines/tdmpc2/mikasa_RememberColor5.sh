# Baseline results for TD-MPC2 (We recommend running individual experiments, instead of the entire file)

seed=(123 321 231)
# Wandb settings 
use_wandb=true
wandb_project="MIKASA-Robo"
wandb_group="tdmpc2"

### State Based TD-MPC2 Baselines ###

## walltime_efficient Setting ##

for seed in ${seed[@]}
do 
    python train.py steps=3_000_000 seed=$seed buffer_size=500_000 exp_name=tdmpc2_mikasa \
        env_id=RememberColor5-v0 num_envs=32 control_mode=pd_ee_delta_pose env_type=gpu obs=rgb \
        save_video_local=true include_joints=true \
        wandb=$use_wandb wandb_project=$wandb_project wandb_group=$wandb_group \
        wandb_name=tdmpc2-RememberColor5-v0-rgb_joint-$seed
done

# for seed in ${seed[@]}
# do 
#     python train.py model_size=5 steps=5_000_000 seed=$seed buffer_size=500_000 exp_name=tdmpc2_mikasa \
#         env_id=RememberColor3-v0 num_envs=32 control_mode=pd_ee_delta_pose env_type=gpu obs=rgb \
#         save_video_local=false include_joints=true \
#         wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project wandb_group=$wandb_group setting_tag=walltime_efficient \
#         wandb_name=tdmpc2-PushCube-v1-state-$seed-walltime_efficient
# done

# python train.py buffer_size=500_000 steps=5_000_000 seed=1 exp_name=default \
#   env_id=ShellGamePush-v0 env_type=gpu num_envs=2 control_mode=pd_ee_delta_pose obs=rgb \
#   save_video_local=false include_joints=true