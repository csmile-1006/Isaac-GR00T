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

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from .cross_attention_dit import DiT, SelfAttentionTransformer
from .flow_matching_action_head import CategorySpecificLinear, CategorySpecificMLP, MultiEmbodimentActionEncoder, swish


def expectile_loss(adv, diff, expectile):
    """Compute the expectile loss."""
    weight = torch.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


class BroNet(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int, add_final_layer: bool = False, output_dim: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.add_final_layer = add_final_layer
        self.output_dim = output_dim

        # Create the residual blocks based on depth
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.residual_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            block = ResidualBlock(hidden_dim)
            self.residual_blocks.append(block)

        if add_final_layer:
            self.final_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layernorm(self.input_projection(x)))
        # Pass through each residual block sequentially
        for block in self.residual_blocks:
            x = block(x)

        if self.add_final_layer:
            x = self.final_layer(x)

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            # First dense block
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # Second dense block
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # Final transformation before residual connection
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        return x + identity


class DoubleCritic(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int, add_final_layer: bool = True, output_dim: int = 1
    ):
        super().__init__()
        self.Q1 = BroNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )
        self.Q2 = BroNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, states, actions):
        B = states.shape[0]
        state_action = torch.cat([states.reshape(B, -1), actions.reshape(B, -1)], axis=1)
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        return q1, q2


class Value(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, depth: int, add_final_layer: bool = True, output_dim: int = 1
    ):
        super().__init__()
        self.value = BroNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, states):
        B = states.shape[0]
        v = self.value(states.reshape(B, -1))
        return v


class MultiEmbodimentActionCriticEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)

    def forward(self, actions, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Standard action MLP step for shape => (B, T, w)
        a_emb = swish(self.W1(actions, cat_ids))

        # 2) W2 => (B, T, w)
        x = self.W2(a_emb, cat_ids)

        return x


@dataclass
class CriticConfig(PretrainedConfig):
    hidden_dim: int = field(default=1024, metadata={"help": "Hidden dimension."})
    depth: int = field(default=2, metadata={"help": "Depth of the network."})
    add_final_layer: bool = field(default=True, metadata={"help": "Whether to add a final layer."})
    output_dim: int = field(default=1, metadata={"help": "Output dimension."})


@dataclass
class DualFlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    backbone_embedding_dim: int = field(default=1536, metadata={"help": "Backbone embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
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
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)

    critic_config: dict = field(init=False, metadata={"help": "Critic model config."})

    # RL parameters
    expectile: float = field(default=0.9, metadata={"help": "Expectile for expectile loss."})
    q_agg: str = field(default="min", metadata={"help": "Aggregation function for critic loss."})
    discount: float = field(default=0.999, metadata={"help": "Discount factor for MDP."})
    nstep: int = field(default=1, metadata={"help": "Number of steps for reward."})
    normalize_q: bool = field(default=False, metadata={"help": "Whether to normalize the Q-value."})
    alpha: float = field(default=10.0, metadata={"help": "Alpha for actor loss."})
    tau: float = field(default=0.005, metadata={"help": "Tau for polyak update."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DualFlowmatchingActionHead(nn.Module):
    config_class = DualFlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: DualFlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.onestep_model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
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
        self.critic_action_encoder = MultiEmbodimentActionCriticEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
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

        self.value = Value(
            input_dim=self.input_embedding_dim,
            hidden_dim=config.critic_config["hidden_dim"],
            depth=config.critic_config["depth"],
            add_final_layer=config.critic_config["add_final_layer"],
            output_dim=config.critic_config["output_dim"],
        )

        self.critic = DoubleCritic(
            input_dim=self.input_embedding_dim * (config.action_horizon + 1),
            hidden_dim=config.critic_config["hidden_dim"],
            depth=config.critic_config["depth"],
            add_final_layer=config.critic_config["add_final_layer"],
            output_dim=config.critic_config["output_dim"],
        )

        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())

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

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

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
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
                self.onestep_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # get predicted action from multi-step diffusion model
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
        state_features = self.state_encoder(action_input.state, embodiment_id)
        action_critic_features = self.critic_action_encoder(action_input.action, embodiment_id)

        # Critic loss 1) Value loss
        with torch.no_grad():
            q1, q2 = self.target_critic(state_features, action_critic_features)
            if self.config.q_agg == "min":
                q = torch.minimum(q1, q2)
            elif self.config.q_agg == "mean":
                q = (q1 + q2) / 2

        v = self.value(state_features)
        value_loss = expectile_loss(q - v, q - v, self.config.expectile).mean()

        # Critic loss 2) critic loss
        next_state_features = self.state_encoder(action_input.next_state, embodiment_id)
        with torch.no_grad():
            next_v = self.value(next_state_features)
            q = (
                action_input.reward
                + (self.config.discount ** (self.config.nstep * self.config.action_horizon)) * action_input.done * next_v
            )
        q1, q2 = self.critic(state_features, action_critic_features)
        critic_loss = ((q - q1) ** 2 + (q - q2) ** 2).mean()

        # Actor loss 1) distillation loss
        # 1-1: Get action trajectory from multi-step diffusion model
        device = vl_embeds.device
        noises = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        vl_embs = vl_embeds
        multistep_actions = noises.detach().clone()
        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            multistep_action_features = self.action_encoder(multistep_actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(multistep_action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                multistep_action_features = multistep_action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, multistep_action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            multistep_actions = multistep_actions + dt * pred_velocity

        # 1-2: Get action trajectory from one-step diffusion model
        onestep_actions = noises.detach().clone()
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=0, device=device)
        onestep_action_features = self.action_encoder(onestep_actions, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(onestep_action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            onestep_action_features = onestep_action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, onestep_action_features), dim=1)
        # Run model forward.
        onestep_model_output = self.onestep_model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        onestep_pred = self.onestep_action_decoder(onestep_model_output, embodiment_id)
        onestep_pred_velocity = onestep_pred[:, -self.action_horizon :]

        onestep_actions = noises + onestep_pred_velocity

        # 1-2: Distillation loss
        distillation_loss = F.mse_loss(multistep_actions, onestep_actions)

        # Actor loss 2) Q-value loss
        actor_action_critic_features = self.critic_action_encoder(onestep_actions, embodiment_id)
        q1, q2 = self.critic(state_features, actor_action_critic_features)
        q = (q1 + q2) / 2
        q_loss = -q.mean()
        if self.config.normalize_q:
            lam = (1 / torch.abs(q).mean()).detach()
            q_loss = lam * q_loss

        actor_loss = q_loss + self.config.alpha * distillation_loss
        total_loss = critic_loss + value_loss + actor_loss

        output_dict = {
            "loss": total_loss,
            "distillation_loss": distillation_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            "actor_loss": actor_loss,
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
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
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

        pred_velocity = pred[:, -self.action_horizon :]

        actions = actions + pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
