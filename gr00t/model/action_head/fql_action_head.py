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

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.critic.critic import DoubleCritic

from .cross_attention_dit import DiT, SelfAttentionTransformer
from .flow_matching_action_head import CategorySpecificMLP, MultiEmbodimentActionEncoder


@dataclass
class CriticConfig(PretrainedConfig):
    hidden_dim: int = field(default=512, metadata={"help": "Hidden dimension."})
    depth: int = field(default=4, metadata={"help": "Depth of the network."})
    output_dim: int = field(default=1, metadata={"help": "Output dimension."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class RLConfig(PretrainedConfig):
    # RL parameters
    critic_action_horizon: int = field(default=1, metadata={"help": "Critic action horizon."})
    q_agg: str = field(default="min", metadata={"help": "Aggregation function for critic loss."})
    discount1: float = field(default=0.99, metadata={"help": "Discount factor for inner MDP."})
    discount2: float = field(default=0.99, metadata={"help": "Discount factor for outer MDP."})
    negative_reward: bool = field(default=True, metadata={"help": "Whether the reward is negative."})
    nstep: int = field(default=1, metadata={"help": "Number of steps for reward."})
    normalize_q: bool = field(default=True, metadata={"help": "Whether to normalize the Q-value."})
    alpha: float = field(default=3.0, metadata={"help": "Alpha for actor loss."})
    tau: float = field(default=0.005, metadata={"help": "Tau for polyak update."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class FQLActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default_factory=dict, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(default=1536, metadata={"help": "Backbone embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=7, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=16, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=4,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default="", metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=1)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default_factory=dict)
    critic_config: dict = field(default_factory=dict)
    rl_config: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FQLActionHead(nn.Module):
    config_class = FQLActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FQLActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.onestep_model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.rl_config = config.rl_config
        self.critic_action_horizon = self.rl_config.get("critic_action_horizon", 1)
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.onestep_action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.critic_action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        self.backbone_encoder = nn.Sequential(
            nn.Linear(config.backbone_embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.input_embedding_dim),
        )

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.onestep_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.critic_config = config.critic_config
        self.critic = DoubleCritic(
            input_dim=self.input_embedding_dim * (self.critic_action_horizon + 2),
            hidden_dims=[self.critic_config["hidden_dim"]] * self.critic_config["depth"],
            output_dim=self.critic_config["output_dim"],
        )

        self.target_critic = DoubleCritic(
            input_dim=self.input_embedding_dim * (self.critic_action_horizon + 2),
            hidden_dims=[self.critic_config["hidden_dim"]] * self.critic_config["depth"],
            output_dim=self.critic_config["output_dim"],
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool, tune_critic: bool = True):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_critic = tune_critic
        for p in self.parameters():
            p.requires_grad = True
        self.target_critic.requires_grad_(False)
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.onestep_action_encoder.requires_grad_(False)
            self.onestep_action_decoder.requires_grad_(False)
            self.critic_action_encoder.requires_grad_(False)
            self.backbone_encoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
            self.onestep_model.requires_grad_(False)
        if not tune_critic:
            self.critic.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head critic: {self.tune_critic}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_critic:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def get_parameter_groups_for_separate_optimizers(self):
        """Get parameter groups for separate optimizers based on loss components."""
        flow_matching_params = []
        actor_params = []
        critic_params = []

        # To avoid parameter overlap between groups, we will:
        # 1. Build a mapping from parameter to group name(s)
        # 2. Only assign each parameter to the first group it appears in (flow_matching > actor > critic)
        # 3. Remove duplicates

        # Collect all parameters for each group (with requires_grad)
        flow_matching_params = []
        actor_params = []
        critic_params = []

        if self.tune_projector:
            flow_matching_params.extend(list(self.state_encoder.parameters()))
            flow_matching_params.extend(list(self.action_encoder.parameters()))
            flow_matching_params.extend(list(self.action_decoder.parameters()))
            if self.config.add_pos_embed:
                flow_matching_params.extend(list(self.position_embedding.parameters()))
            actor_params.extend(list(self.onestep_action_encoder.parameters()))
            actor_params.extend(list(self.onestep_action_decoder.parameters()))

        if self.tune_diffusion_model:
            flow_matching_params.extend(list(self.model.parameters()))
            actor_params.extend(list(self.onestep_model.parameters()))

        if self.tune_critic:
            critic_params.extend(list(self.critic_action_encoder.parameters()))
            critic_params.extend(list(self.critic.parameters()))
            critic_params.extend(list(self.target_critic.parameters()))
            critic_params.extend(list(self.backbone_encoder.parameters()))

        # Only keep parameters that require gradients
        flow_matching_params = [p for p in flow_matching_params if p.requires_grad]
        actor_params = [p for p in actor_params if p.requires_grad]
        critic_params = [p for p in critic_params if p.requires_grad]

        return {"flow_matching": flow_matching_params, "actor": actor_params, "critic": critic_params}
        # if self.freeze_decode_layer:
        #     self.decode_layer.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.onestep_action_encoder.eval()
                self.onestep_action_decoder.eval()
                self.critic_action_encoder.eval()
                self.backbone_encoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
                self.onestep_model.eval()
            if not self.tune_critic:
                self.critic.eval()
                self.target_critic.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def flow_matching_loss(self, backbone_output: BatchFeature, action_input: BatchFeature) -> torch.Tensor:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        return loss

    def compute_flow_matching_loss(self, backbone_output: BatchFeature, action_input: BatchFeature) -> torch.Tensor:
        """Compute flow matching loss with proper gradient isolation."""
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)
        loss = self.flow_matching_loss(backbone_output, action_input)
        return loss

    def compute_actor_loss(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute actor loss (distillation + Q-value) with proper gradient isolation."""
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        batch_size = vl_embeds.shape[0]

        # Embed state.
        with torch.no_grad():
            state_features = self.state_encoder(action_input.state, embodiment_id)

        # Actor loss 1) distillation loss
        device = vl_embeds.device
        noises = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        def run_diffusion(
            initial_actions,
            state_features,
            vl_embs,
            embodiment_id,
            batch_size,
            model,
            action_encoder,
            action_decoder,
            device,
            num_steps=1,
        ):
            """Run diffusion process (multi-step or one-step) to generate actions."""
            actions = initial_actions.detach().clone()
            dt = 1.0 / num_steps

            for t in range(num_steps):
                if num_steps == 1:
                    t_discretized = 0
                else:
                    t_cont = t / float(num_steps)
                    t_discretized = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
                action_features = action_encoder(actions, timesteps_tensor, embodiment_id)
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs
                sa_embs = torch.cat((state_features, action_features), dim=1)
                model_output = model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
                pred = action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -self.action_horizon :]
                actions = actions + dt * pred_velocity
            return actions

        # Multi-step diffusion
        with torch.no_grad():
            multistep_actions = run_diffusion(
                noises,
                state_features,
                vl_embeds,
                embodiment_id,
                batch_size,
                self.model,
                self.action_encoder,
                self.action_decoder,
                device,
                num_steps=self.num_inference_timesteps,
            )

        # One-step diffusion
        onestep_actions = run_diffusion(
            noises,
            state_features,
            vl_embeds,
            embodiment_id,
            batch_size,
            self.onestep_model,
            self.onestep_action_encoder,
            self.onestep_action_decoder,
            device,
            num_steps=1,
        )

        # 1-2: Distillation loss
        distillation_loss = F.mse_loss(multistep_actions, onestep_actions)

        # Actor loss 2) Q-value loss
        vl_embeds_mean = vl_embeds.mean(dim=1)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean)
        timestep_tensor = torch.full(size=(batch_size,), fill_value=0, device=device)
        actor_action_critic_features = self.critic_action_encoder(
            onestep_actions[:, :self.critic_action_horizon], timestep_tensor, embodiment_id
        )
        q1, q2 = self.critic(vl_embed_features, state_features, actor_action_critic_features)
        q = (q1 + q2) / 2
        q_loss = -q.mean()
        if self.rl_config.get("normalize_q", False):
            lam = (1 / torch.abs(q).mean()).detach()
            q_loss = lam * q_loss

        actor_loss = q_loss + self.rl_config.get("alpha", 1.0) * distillation_loss
        metrics = {
            "q_loss": q_loss.detach(),
            "q_mean": q.detach().mean(),
            "q_std": q.detach().std(),
            "q_min": q.detach().min(),
            "q_max": q.detach().max(),
            "mse": F.mse_loss(onestep_actions, multistep_actions).detach(),
        }
        return actor_loss, distillation_loss, metrics

    def compute_critic_loss(
        self, backbone_output: BatchFeature, next_backbone_output: BatchFeature, action_input: BatchFeature
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute critic loss with proper gradient isolation."""
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)
        next_backbone_output = self.process_backbone_output(next_backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

            for k, v in next_backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                next_backbone_output[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        next_vl_embeds = next_backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        # Embed state.
        with torch.no_grad():
            state_features = self.state_encoder(action_input.state, embodiment_id)
        timestep_tensor = torch.full(size=(batch_size,), fill_value=0, device=device)
        action_critic_features = self.critic_action_encoder(
            action_input.action[:, :self.critic_action_horizon], timestep_tensor, embodiment_id
        )
        vl_embeds_mean = vl_embeds.mean(dim=1)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean)

        # Critic loss
        next_state_features = self.state_encoder(action_input.next_state, embodiment_id)

        done = torch.prod(action_input.done, dim=-1)
        reward = action_input.reward
        if self.rl_config.get("negative_reward", False):
            reward -= 1
        discounts1 = self.rl_config.get("discount1", 0.99) ** torch.arange(self.critic_action_horizon).to(reward.device)
        scaled_rewards = torch.sum(reward * discounts1, dim=-1)

        with torch.no_grad():
            next_action_input = BatchFeature(data={"state": action_input.next_state, "embodiment_id": embodiment_id})
            next_pred_actions = self.get_action(next_backbone_output, next_action_input)["action_pred"]
            next_vl_embeds_mean = next_vl_embeds.mean(dim=1)
            next_vl_embed_features = self.backbone_encoder(next_vl_embeds_mean)
            next_action_critic_features = self.critic_action_encoder(
                next_pred_actions[:, :self.critic_action_horizon], timestep_tensor, embodiment_id
            )
            next_q1, next_q2 = self.target_critic(
                next_vl_embed_features, next_state_features, next_action_critic_features
            )
            if self.rl_config.get("q_agg", "min") == "min":
                next_q = torch.minimum(next_q1, next_q2)
            elif self.rl_config.get("q_agg", "min") == "mean":
                next_q = (next_q1 + next_q2) / 2
            else:
                assert False, f"Invalid q_agg: {self.rl_config.get('q_agg', 'min')}"

            target_q = (
                scaled_rewards
                + (self.rl_config.get("discount2", 0.99) ** (self.rl_config.get("nstep", 1) * self.critic_action_horizon))
                * (1. - done)
                * next_q
            )
        q1, q2 = self.critic(vl_embed_features, state_features, action_critic_features)
        critic_loss = ((target_q - q1) ** 2 + (target_q - q2) ** 2).mean()

        next_q_val = next_q.detach()
        target_q_val = target_q.detach()
        q1_val = q1.detach()
        q2_val = q2.detach()
        metrics = {
            "target_q_mean": target_q_val.mean(),
            "target_q_std": target_q_val.std(),
            "target_q_min": target_q_val.min(),
            "target_q_max": target_q_val.max(),
            "next_q_mean": next_q_val.mean(),
            "next_q_std": next_q_val.std(),
            "next_q_min": next_q_val.min(),
            "next_q_max": next_q_val.max(),
            "q1_mean": q1_val.mean(),
            "q1_std": q1_val.std(),
            "q1_min": q1_val.min(),
            "q1_max": q1.max(),
            "q2_mean": q2_val.mean(),
            "q2_std": q2_val.std(),
            "q2_min": q2_val.min(),
            "q2_max": q2_val.max(),
            "batch_reward": scaled_rewards.detach().mean(),
        }
        return critic_loss, metrics

    def forward(
        self, backbone_output: BatchFeature, next_backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        # Compute each loss separately to avoid gradient conflicts
        flow_matching_loss = self.compute_flow_matching_loss(backbone_output, action_input)
        actor_loss, distillation_loss, actor_metrics = self.compute_actor_loss(backbone_output, action_input)
        critic_loss, critic_metrics = self.compute_critic_loss(backbone_output, next_backbone_output, action_input)

        total_loss = flow_matching_loss + critic_loss + actor_loss

        output_dict = {
            "loss": total_loss,
            "flow_matching_loss": flow_matching_loss,
            "distillation_loss": distillation_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            **{f"actor/{k}": v for k, v in actor_metrics.items()},
            **{f"critic/{k}": v for k, v in critic_metrics.items()},
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        # Run denoising steps.
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=0, device=device)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        vl_embs = vl_embeds

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # Run model forward.
        model_output = self.onestep_model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output, embodiment_id)

        pred_velocity = pred[:, :self.critic_action_horizon]

        actions = actions + pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
