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

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.our_state_critic import (
    OurStateCritic,
    OurStateCriticConfig,
)
from .gr00t_n1 import GR00T_N1_5

ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Ours_State_Critic_Config(PretrainedConfig):
    model_type = "gr00t_n1_5_ours_critic"

    critic_cfg: dict = field(init=False, metadata={"help": "Critic configuration."})

    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5_Ours_State_Critic(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Ours_State_Critic_Config
    """
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Ours_State_Critic_Config,
        local_model_path: str,
    ):
        assert isinstance(config.critic_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        critic_cfg = OurStateCriticConfig(**config.critic_cfg)
        self.critic_head = OurStateCritic(critic_cfg)

        self.critic_action_horizon = critic_cfg.rl_config["critic_action_horizon"]
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3 and action.shape[1] == self.action_horizon and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, critic_outputs, is_training=True):
        fail_critic = (not isinstance(critic_outputs, BatchFeature)) or not (
            (LOSS_KEY in critic_outputs and is_training)  # there might not be an action prediction during training
        )

        if fail_critic:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(critic_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in critic_outputs=}"
            error_msg += f"\n{critic_outputs[LOSS_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        action_inputs = self.prepare_input(inputs)
        critic_outputs = self.critic_head(action_inputs)
        self.validate_data(critic_outputs, is_training=True)
        return critic_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature]:
        self.validate_inputs(inputs)
        action_inputs = self.critic_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.critic_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return action_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        value_cfg: Optional[dict] = None,
        critic_cfg: Optional[dict] = None,
        rl_cfg: Optional[dict] = None,
        from_gr00t_n1_5: bool = False,
        **kwargs,
    ):
        tune_critic = kwargs.pop("tune_critic", True)
        tune_value = kwargs.pop("tune_value", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune action head critic: {tune_critic}")
        print(f"Tune action head value: {tune_value}")

        if not from_gr00t_n1_5:
            # get the current model path being downloaded
            try:
                # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
                # saved in ~/.cache/huggingface/hub/
                local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
                # HFValidationError, RepositoryNotFoundError
            except (HFValidationError, RepositoryNotFoundError):
                print(
                    f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
                )
                local_model_path = pretrained_model_name_or_path

            pretrained_model = super().from_pretrained(local_model_path, local_model_path=local_model_path, **kwargs)

            pretrained_model.critic_head.set_trainable_parameters(
                tune_value=tune_value,
                tune_critic=tune_critic,
            )
            return pretrained_model

        else:
            pretrained_gr00t_n1_5 = GR00T_N1_5.from_pretrained(pretrained_model_name_or_path, **kwargs)

            new_cfg = GR00T_N1_5_Ours_State_Critic_Config()
            pretrained_gr00t_n1_5_cfg = pretrained_gr00t_n1_5.config.to_dict()
            for key, value in pretrained_gr00t_n1_5_cfg.items():
                if key != "action_head_cfg":
                    setattr(new_cfg, key, value)

            # Transfer action head config
            critic_cfg = OurStateCriticConfig(**pretrained_gr00t_n1_5_cfg["action_head_cfg"])
            critic_cfg.critic_config = critic_cfg
            critic_cfg.value_config = value_cfg
            critic_cfg.rl_config = rl_cfg
            new_cfg.critic_cfg = critic_cfg.to_dict()

            pretrained_model = cls(
                config=new_cfg,
                local_model_path=pretrained_gr00t_n1_5.local_model_path,
            )

            # Transfer parameters from pretrained GR00T_N1_5 model

            # Transfer action head parameters
            critic_components = {
                "state_encoder": "state_encoder",
                "ca_encoder": "action_encoder",  # Uses same model
            }

            with torch.no_grad():
                full_src = pretrained_gr00t_n1_5.action_head.state_dict()

                for comp_name, prefix in tqdm(critic_components.items(), desc="Loading action head parameters"):
                    subdict = {k[len(prefix) + 1 :]: v for k, v in full_src.items() if k.startswith(prefix)}
                    getattr(pretrained_model.critic_head, comp_name).load_state_dict(subdict)

            # Set trainable parameters according to flags
            pretrained_model.critic_head.set_trainable_parameters(
                tune_critic=tune_critic,
                tune_value=tune_value,
            )

            del pretrained_gr00t_n1_5
            return pretrained_model


# register
AutoConfig.register("gr00t_n1_5_ours_critic", GR00T_N1_5_Ours_State_Critic_Config)
AutoModel.register(GR00T_N1_5_Ours_State_Critic_Config, GR00T_N1_5_Ours_State_Critic)
