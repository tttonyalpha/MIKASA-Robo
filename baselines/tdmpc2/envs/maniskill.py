import gymnasium as gym
import numpy as np
from common.logger import Logger
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.record_episode import RecordEpisodeWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
# from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from mani_skill.utils import gym_utils
from functools import partial
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

import mani_skill.envs

from mikasa_robo_suite.memory_envs import *
from mikasa_robo_suite.utils.wrappers import *

import copy
from typing import Dict
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from tqdm import tqdm

from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state".

    Args:
        rgb (bool): Whether to include rgb images in the observation.
        depth (bool): Whether to include depth images in the observation.
        state (bool): Whether to include state data in the observation.
    """

    def __init__(self, env, rgb=True, depth=True, state=True, oracle=False,
                 joints=False) -> None:
        self.base_env: BaseEnv = StateOnlyTensorToDictWrapper(env.unwrapped)
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_oracle = oracle
        self.include_joints = joints

        sample_obs, _ = env.reset()
        new_obs = self.observation(sample_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        ret = dict()

        if self.include_rgb or self.include_depth:
            ret['oracle_info'] = observation['oracle_info']
            ret['prompt'] = observation['prompt']
            sensor_data = observation.pop("sensor_data")
            del observation["sensor_param"]

            images = []
            for cam_data in sensor_data.values():
                if self.include_rgb:
                    images.append(cam_data["rgb"])
                if self.include_depth:
                    images.append(cam_data["depth"])

            if len(images) > 0:
                images = torch.concat(images, axis=-1)

        # Flatten the rest of the data which should just be state data
        if self.include_state and not (self.include_rgb or self.include_depth):
            if not self.include_oracle:
                observation.pop("oracle_info")
            else:
                observation = observation
        else:
            if not self.include_joints:
                filtered_obs = {
                    k: v for k, v in observation.items()
                    if k not in ['prompt', 'oracle_info']
                }
            else:
                # Create extra_agent dict with 'extra' and 'agent' keys
                extra_agent = {}
                for key in ['extra', 'agent']:
                    if key in observation:
                        extra_agent[key] = observation.pop(key)

                # Flatten the extra_agent dict
                extra_agent_flat = common.flatten_state_dict(
                    extra_agent, use_torch=True, device=self.base_env.device
                )
                ret['joints'] = extra_agent_flat

                filtered_obs = {
                    k: v for k, v in observation.items()
                    if k not in ['prompt', 'oracle_info', 'extra']
                }

            observation = common.flatten_state_dict(
                filtered_obs, use_torch=True, device=self.base_env.device
            )

        if self.include_state and not (self.include_rgb or self.include_depth):
            ret = observation
        else:
            ret["state"] = observation

        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images

        if 'state' in ret.keys() and not self.include_state:
            ret.pop('state')

        if ('oracle_info' in ret.keys() and not self.include_oracle and
                ret['oracle_info'] is not None):
            ret.pop('oracle_info')

        if ('oracle_info' in ret.keys() and
                (ret['oracle_info'] == 4242424242).any().item()):
            ret.pop('oracle_info')

        if ('prompt' in ret.keys() and
                (ret['prompt'] == 4242424242).any().item()):
            ret.pop('prompt')

        if 'joints' in ret.keys() and not self.include_joints:
            ret.pop('joints')

        return ret


def cpu_env_factory(
    env_make_fn, idx: int, wrappers=[], record_video_path: str = None,
    record_episode_kwargs=dict(), logger: Logger = None
):
    def _init():
        env = env_make_fn()
        for wrapper in wrappers:
            env = wrapper(env)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if record_video_path is not None and (
            not record_episode_kwargs["record_single"] or idx == 0
        ):
            env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                info_on_video=record_episode_kwargs["info_on_video"],
                logger=logger,
            )
        return env

    return _init


def make_envs(cfg, num_envs, record_video_path, is_eval, logger):
    """
    Make ManiSkill3 environment.
    """
    record_episode_kwargs = dict(
        save_video=True,
        save_trajectory=False,
        record_single=True,
        info_on_video=False,
    )

    # Set up env make function for consistency
    env_make_fn = partial(
        gym.make,
        disable_env_checker=True,
        id=cfg.env_id,
        obs_mode=cfg.obs,
        render_mode=cfg.render_mode,
        sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
    )
    TIME = time.strftime('%Y%m%d_%H%M%S')

    if cfg.env_id in ['ShellGamePush-v0', 'ShellGamePick-v0', 'ShellGameTouch-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'cup_with_ball_number'
        prompt_info = None
    elif cfg.env_id in [
        'InterceptSlow-v0', 'InterceptMedium-v0', 'InterceptFast-v0',
        'InterceptGrabSlow-v0', 'InterceptGrabMedium-v0', 'InterceptGrabFast-v0',
    ]:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in [
        'RotateLenientPos-v0', 'RotateLenientPosNeg-v0',
        'RotateStrictPos-v0', 'RotateStrictPosNeg-v0',
    ]:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = 'angle_diff'
        prompt_info = 'target_angle'
    elif cfg.env_id in ['CameraShutdownPush-v0', 'CameraShutdownPick-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (CameraShutdownWrapper, {"n_initial_steps": 19}),  # camera works only for t ~ [0, 19]
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['TakeItBack-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['RememberColor3-v0', 'RememberColor5-v0', 'RememberColor9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['RememberShape3-v0', 'RememberShape5-v0', 'RememberShape9-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in [
        'RememberShapeAndColor3x2-v0', 'RememberShapeAndColor3x3-v0',
        'RememberShapeAndColor5x3-v0',
    ]:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['BunchOfColors3-v0', 'BunchOfColors5-v0', 'BunchOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['SeqOfColors3-v0', 'SeqOfColors5-v0', 'SeqOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    elif cfg.env_id in ['ChainOfColors3-v0', 'ChainOfColors5-v0', 'ChainOfColors7-v0']:
        wrappers_list = [
            (InitialZeroActionWrapper, {"n_initial_steps": cfg.noop_steps - 1}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
        oracle_info = None
        prompt_info = None
    else:
        raise ValueError(f"Unknown environment: {cfg.env_id}")

    print('\n' + '=' * 75)
    print('║' + ' ' * 24 + 'Environment Configuration' + ' ' * 24 + '║')
    print('=' * 75)
    print('║' + f' Environment ID: {cfg.env_id}'.ljust(73) + '║')
    print('║' + f' Oracle Info:    {oracle_info}'.ljust(73) + '║')
    print('║ Wrappers:'.ljust(74) + '║')
    for wrapper, kwargs in wrappers_list:
        print('║    ├─ ' + wrapper.__name__.ljust(65) + '║')
        if kwargs:
            print('║    │  └─ ' + str(kwargs).ljust(65) + '║')
    print('║' + '-' * 73 + '║')

    state_msg = 'state will be used' if cfg.include_state else 'state will not be used'
    print('║' + f' include_state:       {str(cfg.include_state):<5} │ {state_msg}'.ljust(68) + '║')

    rgb_msg = 'rgb images will be used' if cfg.include_rgb else 'rgb images will not be used'
    print('║' + f' include_rgb:         {str(cfg.include_rgb):<5} │ {rgb_msg}'.ljust(68) + '║')

    oracle_msg = (
        'oracle info will be used'
        if cfg.include_oracle else 'oracle info will not be used'
    )
    print('║' + f' include_oracle:      {str(cfg.include_oracle):<5} │ {oracle_msg}'.ljust(68) + '║')

    joints_msg = 'joints will be used' if cfg.include_joints else 'joints will not be used'
    print('║' + f' include_joints:      {str(cfg.include_joints):<5} │ {joints_msg}'.ljust(68) + '║')
    print('=' * 75 + '\n')

    assert any([cfg.include_state, cfg.include_rgb]), (
        "At least one of include_state or include_rgb must be True."
    )
    assert not (cfg.include_joints and not cfg.include_rgb), (
        "include_joints can only be True when include_rgb is True"
    )

    if cfg.include_state and not cfg.include_rgb and not cfg.include_oracle and not cfg.include_joints:
        MODE = 'state'
    elif cfg.include_state and cfg.include_rgb and not cfg.include_oracle and not cfg.include_joints:
        raise NotImplementedError(
            "state_rgb is not implemented and does not make sense, since any environment "
            "can be solved only by using state"
        )
        MODE = 'state_rgb'
    elif cfg.include_state and not cfg.include_rgb and cfg.include_oracle and not cfg.include_joints:
        raise NotImplementedError(
            "state_oracle is not implemented and does not make sense, since the state already "
            "contains oracle information"
        )
        MODE = 'state_oracle'
    elif cfg.include_state and cfg.include_rgb and cfg.include_oracle and not cfg.include_joints:
        raise NotImplementedError(
            "state_rgb_oracle is not implemented and does not make sense, since any environment "
            "can be solved only by using state"
        )
        MODE = 'state_rgb_oracle'
    elif not cfg.include_state and cfg.include_rgb and not cfg.include_oracle and not cfg.include_joints:
        MODE = 'rgb'
    elif not cfg.include_state and cfg.include_rgb and cfg.include_oracle and not cfg.include_joints:
        MODE = 'rgb_oracle'
    elif not cfg.include_state and cfg.include_rgb and cfg.include_joints and cfg.include_oracle:
        MODE = 'rgb_joints_oracle'  # TODO: check if this is correct
    elif not cfg.include_state and cfg.include_rgb and cfg.include_joints and not cfg.include_oracle:
        MODE = 'rgb_joints'
    else:
        raise NotImplementedError(
            f"Unknown mode: {cfg.include_state=} {cfg.include_rgb=} "
            f"{cfg.include_oracle=} {cfg.include_joints=}"
        )

    SAVE_DIR = f'checkpoints/ppo_memtasks/{MODE}/{cfg.reward_mode}/{cfg.env_id}'

    print(f'{MODE=}')
    print(f'{prompt_info=}')

    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {}))
    # obs=torch.tensor -> dict with keys: state: obs, prompt: prompt, oracle_info: oracle_info

    if cfg.control_mode != 'default':
        env_make_fn = partial(env_make_fn, control_mode=cfg.control_mode)
    if is_eval:
        # https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation
        env_make_fn = partial(env_make_fn, reconfiguration_freq=cfg.eval_reconfiguration_frequency)

    if cfg.env_type == 'cpu':
        # Get default control_mode and max_episode_steps values
        dummy_env = env_make_fn()
        control_mode = dummy_env.control_mode
        max_episode_steps = gym_utils.find_max_episode_steps_value(dummy_env)
        dummy_env.close()
        del dummy_env
        # Create cpu async vectorized env
        vector_env_cls = partial(AsyncVectorEnv, context="forkserver")
        if num_envs == 1:
            vector_env_cls = SyncVectorEnv
        wrappers = []
        if cfg['obs'] == 'rgb':
            wrappers.append(partial(PixelWrapper, cfg=cfg, num_envs=num_envs))
        env: VectorEnv = vector_env_cls(
            [
                cpu_env_factory(
                    env_make_fn, i, wrappers, record_video_path, record_episode_kwargs, logger
                )
                for i in range(num_envs)
            ]
        )
        env = TensorWrapper(env)
    elif cfg.env_type == 'gpu':
        env = env_make_fn(num_envs=num_envs)
        for wrapper_class, wrapper_kwargs in wrappers_list:
            env = wrapper_class(env, **wrapper_kwargs)
        control_mode = env.control_mode
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        if cfg['obs'] == 'rgb':
            env = FlattenRGBDObservationWrapper(
                env,
                rgb=True,
                depth=False,
                joints=cfg.include_joints,
                state=cfg.include_state,
            )
            env = PixelWrapper(cfg, env, num_envs)
        if record_video_path is not None:
            env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name="trajectory",
                max_steps_per_video=max_episode_steps,
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                logger=logger,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    else:
        raise Exception('env_type must be cpu or gpu')

    cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = control_mode
    cfg.env_cfg.env_horizon = cfg.eval_env_cfg.env_horizon = env.max_episode_steps = max_episode_steps

    return env
