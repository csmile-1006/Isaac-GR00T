import gymnasium as gym
import numpy as np
from robocasa.utils.env_utils import create_env
from robosuite.wrappers import GymWrapper


def load_robocasa_gym_env(env_name, seed=None):
    env = create_env(
        env_name=env_name,
        render_onscreen=False,
        seed=seed,  # set seed=None to run unseeded
        camera_widths=256,
        camera_heights=256,
    )
    env = GymWrapper(
        env,
        flatten_obs=False,
        keys=[
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_eef_pos",
            "robot0_base_to_eef_pos",
            "robot0_eef_quat",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
        ],
    )
    return env


class RoboCasaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._robocasa_keys_to_gr00t_keys = {
            "robot0_base_pos": "state.base_position",
            "robot0_base_quat": "state.base_rotation",
            "robot0_eef_pos": "state.end_effector_position_absolute",
            "robot0_base_to_eef_pos": "state.end_effector_position_relative",
            "robot0_eef_quat": "state.end_effector_rotation_absolute",
            "robot0_base_to_eef_quat": "state.end_effector_rotation_relative",
            "robot0_gripper_qpos": "state.gripper_qpos",
            "robot0_gripper_qvel": "state.gripper_qvel",
            "robot0_joint_pos": "state.joint_position",
            "robot0_joint_pos_cos": "state.joint_position_cos",
            "robot0_joint_pos_sin": "state.joint_position_sin",
            "robot0_joint_vel": "state.joint_velocity",
            "robot0_agentview_left_image": "video.left_view",
            "robot0_agentview_right_image": "video.right_view",
            "robot0_eye_in_hand_image": "video.wrist_view",
        }

        self._observation_space = self._convert_observation_space()
        self._action_space = self._convert_action_space()

    def _convert_action_space(self):
        original_action_space = self.env.action_space
        # Split original action space into parts
        low = original_action_space.low
        high = original_action_space.high
        dtype = original_action_space.dtype

        new_action_space = gym.spaces.Dict(
            {
                "action.end_effector_position": gym.spaces.Box(low=low[0:3], high=high[0:3], dtype=dtype),
                "action.end_effector_rotation": gym.spaces.Box(low=low[3:6], high=high[3:6], dtype=dtype),
                "action.gripper_close": gym.spaces.Box(low=low[6:7], high=high[6:7], dtype=np.int64),
                "action.base_motion": gym.spaces.Box(low=low[7:11], high=high[7:11], dtype=dtype),
                "action.control_mode": gym.spaces.Box(low=low[11:12], high=high[11:12], dtype=np.int64),
            }
        )
        self.action_space_keys = [
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        ]
        return new_action_space

    def _convert_observation_space(self):
        original_observation_space = self.env.observation_space
        new_observation_space = {}
        for key, value in original_observation_space.items():
            if key in self._robocasa_keys_to_gr00t_keys:
                new_observation_space[self._robocasa_keys_to_gr00t_keys[key]] = value
        new_observation_space["annotation.human.action.task_description"] = gym.spaces.Sequence(
            gym.spaces.Text(max_length=1000)
        )
        return gym.spaces.Dict(new_observation_space)

    @property
    def language_instruction(self):
        return self.env.get_ep_meta()["lang"]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        new_obs = {}
        for key, value in obs.items():
            if key in self._robocasa_keys_to_gr00t_keys:
                new_obs[self._robocasa_keys_to_gr00t_keys[key]] = value
        new_obs["annotation.human.action.task_description"] = [self.language_instruction]
        info["is_success"] = self.is_success()["task"]
        return new_obs, info

    def render(self, mode="rgb_array"):
        return self.env.unwrapped.sim.render(camera_name="robot0_agentview_center", height=512, width=512)[::-1]

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return {"task": succ}

    def convert_action(self, action):
        # binarize the gripper close and control mode action
        for key in ["action.gripper_close", "action.control_mode"]:
            action[key] = np.where(action[key] > 0, 1, -1)
        elems = [
            action[key] for key in self.action_space_keys
        ]  # this is to strictly follow the order of the action space
        return np.concatenate(elems, axis=-1)

    def step(self, action):
        action = self.convert_action(action)
        obs, reward, terminated, truncated, info = super().step(action)
        new_obs = {}
        for key, value in obs.items():
            if key in self._robocasa_keys_to_gr00t_keys:
                new_obs[self._robocasa_keys_to_gr00t_keys[key]] = value
        new_obs["annotation.human.action.task_description"] = [self.language_instruction]
        info["is_success"] = self.is_success()["task"]
        terminated = terminated or info["is_success"]
        return new_obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env_name = "PnPCounterToMicrowave"

    env = load_robocasa_gym_env(env_name)
    env = RoboCasaWrapper(env)

    obs, _ = env.reset()

    for i in range(10):
        print(f"Step {i}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"info at step {i}: {info}")
