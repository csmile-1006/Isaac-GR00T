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
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from tqdm import tqdm

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class LeRobotConfig:
    """Configuration for LeRobot dataset settings."""

    data_name: str = "lerobot/robot_sim.PickNPlace"
    output_dir: str = "./lerobot_dataset"
    fps: float = 20.0
    robot_type: str = "GR1ArmsAndWaistFourierHands"


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    lerobot: LeRobotConfig = field(default_factory=LeRobotConfig)


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop("video.ego_view_bg_crop_pad_res256_freq20")
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def create_lerobot_dataset(self, config: SimulationConfig) -> LeRobotDataset:
        """Create a LeRobot dataset from the simulation data."""
        return LeRobotDataset.create(
            repo_id=config.lerobot.data_name,
            root=config.lerobot.output_dir,
            robot_type=config.lerobot.robot_type,
            fps=int(config.lerobot.fps),
            features={
                "observation.images.ego_view": {
                    "dtype": "video",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": (44,),
                },
                "action": {
                    "dtype": "float64",
                    "shape": (44,),
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": (1,),
                },
                "next.reward": {
                    "dtype": "float64",
                    "shape": (1,),
                },
                "annotation.human.coarse_action": {
                    "dtype": "int64",
                    "shape": (1,),
                },
                "annotation.human.fine_action": {
                    "dtype": "int64",
                    "shape": (1,),
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def _convert_dict_to_array(self, dict: Dict[str, Any], env_idx: int, i: int, key: str = "state") -> np.ndarray:
        """Convert a dictionary to a numpy array."""
        target_dict = {
            f"{key}.left_arm": dict[f"{key}.left_arm"][env_idx, i],
            f"{key}.left_hand": dict[f"{key}.left_hand"][env_idx, i],
            f"{key}.left_leg": np.zeros((6,), dtype=np.float64),
            f"{key}.neck": np.zeros((3,), dtype=np.float64),
            f"{key}.right_arm": dict[f"{key}.right_arm"][env_idx, i],
            f"{key}.right_hand": dict[f"{key}.right_hand"][env_idx, i],
            f"{key}.right_leg": np.zeros((6,), dtype=np.float64),
            f"{key}.waist": dict[f"{key}.waist"][env_idx, i],
        }
        return np.concatenate([target_dict[key] for key in target_dict.keys()])

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        dataset = self.create_lerobot_dataset(config)
        print(f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments")
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []

        # Episode data collection: track data for each environment
        episode_data = [[] for _ in range(config.n_envs)]

        # Initial environment reset
        obs, _ = self.env.reset()
        pbar = tqdm(
            total=config.n_episodes,
            desc=f"Evaluating {config.n_episodes} episodes",
            leave=False,
        )
        pbar2 = tqdm(
            total=config.multistep.max_episode_steps,
            desc=f"Evaluating {config.multistep.max_episode_steps} steps",
            leave=False,
        )
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the server
            actions = self._get_actions_from_server(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)

            # Collect episode data for each environment
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1

                # Collect data for this step
                chunk_length = min(config.multistep.n_action_steps, env_infos["rewards"][env_idx].shape[0])
                for i in range(chunk_length):
                    step_data = {
                        "observation.images.ego_view": env_infos["observations"][
                            "video.ego_view_bg_crop_pad_res256_freq20"
                        ][env_idx, i],
                        "observation.state": self._convert_dict_to_array(
                            env_infos["observations"], env_idx, i, key="state"
                        ),
                        "action": self._convert_dict_to_array(actions, env_idx, i, key="action"),
                        "next.reward": np.array([env_infos["rewards"][env_idx][i]], dtype=np.float64),
                        "next.done": np.array(
                            [bool(env_infos["dones"][env_idx][i] or env_infos["truncateds"][env_idx][i])],
                            dtype=bool,
                        ),
                        "annotation.human.coarse_action": np.array([1]),
                        "annotation.human.fine_action": np.array([1]),
                        "task": obs["annotation.human.coarse_action"][env_idx],
                    }
                    episode_data[env_idx].append(step_data)

                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    # Prepare episode data for saving

                    # For success demos, truncate at first success
                    if current_successes[env_idx]:
                        # Find first success index
                        first_success_idx = None
                        for idx, step_data in enumerate(episode_data[env_idx]):
                            if step_data["next.reward"] > 0:
                                first_success_idx = idx
                                break

                        if first_success_idx is not None:
                            # Truncate to include one step after success (success_idx + 1)
                            truncate_idx = first_success_idx + 1
                            episode_data[env_idx] = episode_data[env_idx][:truncate_idx]

                            episode_data[env_idx][-1]["next.done"] = np.array([True])
                            episode_data[env_idx][-1]["next.reward"] = np.array([1.0])

                    for step_data in episode_data[env_idx]:
                        dataset.add_frame(step_data)
                    dataset.save_episode()

                    # Store results
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1

                    # Clear episode data for this environment
                    episode_data[env_idx] = []

                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    pbar.update(1)
                    pbar2.close()
                    pbar2 = tqdm(
                        total=config.multistep.max_episode_steps,
                        desc=f"Evaluating {config.multistep.max_episode_steps} steps",
                        leave=False,
                    )
            obs = next_obs
            pbar2.update(actions[list(actions.keys())[0]].shape[1])
        # Clean up
        pbar.close()
        self.env.reset()
        self.env.close()
        self.env = None
        print(f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds")
        assert len(episode_successes) >= config.n_episodes, (
            f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        )
        return config.env_name, episode_successes

    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(n_action_steps=n_action_steps, max_episode_steps=max_episode_steps),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
