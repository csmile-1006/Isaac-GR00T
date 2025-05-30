# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm, trange

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

warnings.simplefilter("ignore", category=FutureWarning)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


"""
Example command:

python scripts/eval_policy_robocasa.py --host localhost --port 5555
    --action_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
    --env_name CloseDrawer
    --num_episodes 10
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_only",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )

    # robocasa env and evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CloseDrawer",
        help="Name of the robocasa environment to load",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=300,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video",
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)

    # load robocasa env
    env = load_robocasa_gym_env(args.env_name, seed=args.seed)
    env = RoboCasaWrapper(env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    record_video = args.video_path is not None
    if record_video:
        video_base_path = Path(args.video_path)
        # video_base_path.mkdir(parents=True, exist_ok=True)
        episode_trigger = lambda t: t % 1 == 0  # noqa
        env = RecordVideo(env, video_base_path, disable_logger=True, episode_trigger=episode_trigger, fps=20)

    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=args.action_horizon,
    )

    # postprocess function of action, to handle the case where number of dimensions are not the same
    def postprocess_action(action):
        new_action = {}
        for k, v in action.items():
            if v.ndim == 1:
                new_action[k] = v[..., None]
            else:
                new_action[k] = v
        return new_action

    # main evaluation loop
    stats = defaultdict(list)
    for i in trange(args.num_episodes):
        pbar = tqdm(
            total=args.max_episode_steps, desc=f"Episode {i + 1} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
        )
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = policy.get_action(obs)
            post_action = postprocess_action(action)
            next_obs, reward, terminated, truncated, info = env.step(post_action)
            done = terminated or truncated
            step += args.action_horizon
            obs = next_obs
            pbar.update(args.action_horizon)
        add_to(stats, flatten({"is_success": info["is_success"]}))
        pbar.close()

    env.close()

    for k, v in stats.items():
        stats[k] = np.mean(v)
    print(stats)

    exit()
