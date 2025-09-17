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

from gr00t.model.critic.hlg import HLGaussLoss
from gr00t.model.critic.networks import DoubleCritic, Value

from .cross_attention_dit import DiT, SelfAttentionTransformer
from .flow_matching_action_head import CategorySpecificLinear, MultiEmbodimentActionEncoder
from .flow_matching_action_head import CategorySpecificMLP as CategorySpecificMLP_MF


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
    tau: float = field(default=0.005, metadata={"help": "Tau for polyak update."})

    feature_dim: int = field(default=64, metadata={"help": "Feature dimension for using in the critic."})

    num_atoms: int = field(default=101, metadata={"help": "Number of atoms for the critic."})
    sigma: float = field(default=0.1, metadata={"help": "Sigma for the critic."})
    expectile: float = field(default=0.9, metadata={"help": "Expectile for value loss."})
    support_type: str = field(default="geometric", metadata={"help": "Support type for the critic."})

    num_samples: int = field(default=1, metadata={"help": "Number of samples for BoN sampling."})
    temperature: float = field(default=0.0, metadata={"help": "Temperature for BoN sampling."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class OurActionHeadBoNConfig(PretrainedConfig):
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
    tune_critic: bool = field(default=True, metadata={"help": "Whether to tune the critic."})
    tune_value: bool = field(default=True, metadata={"help": "Whether to tune the value."})
    load_pretrained_det_decode_layer_path: str = field(
        default="", metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=1)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default_factory=dict)
    critic_config: dict = field(default_factory=dict)
    value_config: dict = field(default_factory=dict)
    rl_config: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, hidden_dim)
        self.layer3 = CategorySpecificLinear(num_categories, hidden_dim, hidden_dim)
        self.layer4 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.silu(self.layer1(x, cat_ids))
        hidden = F.silu(self.layer2(hidden, cat_ids))
        hidden = F.silu(self.layer3(hidden, cat_ids))
        return self.layer4(hidden, cat_ids)


class OurActionHeadBoN(nn.Module):
    config_class = OurActionHeadBoNConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: OurActionHeadBoNConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.rl_config = RLConfig(**config.rl_config)
        self.critic_action_horizon = self.rl_config.critic_action_horizon
        self.feature_dim = self.rl_config.feature_dim
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP_MF(
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
        self.action_decoder = CategorySpecificMLP_MF(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.backbone_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.backbone_embedding_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.feature_dim,
        )

        self.value_config = CriticConfig(**config.value_config)
        self.value = Value(
            input_dim=config.max_state_dim + self.feature_dim,
            hidden_size=self.value_config.hidden_dim,
            depth=self.value_config.depth,
            output_dim=self.rl_config.num_atoms,
        )

        self.critic_config = CriticConfig(**config.critic_config)
        self.critic = DoubleCritic(
            input_dim=config.max_state_dim + self.feature_dim + self.critic_action_horizon * config.action_dim,
            hidden_dims=[self.critic_config.hidden_dim] * self.critic_config.depth,
            output_dim=self.rl_config.num_atoms,
        )

        self.target_critic = DoubleCritic(
            input_dim=config.max_state_dim + self.feature_dim + self.critic_action_horizon * config.action_dim,
            hidden_dims=[self.critic_config.hidden_dim] * self.critic_config.depth,
            output_dim=self.rl_config.num_atoms,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        # compute v_min and v_max according to the discount factor
        if self.rl_config.negative_reward:
            v_min = -1 * (1 / (1 - self.rl_config.discount2))
            v_max = 0.0
        else:
            v_min = 0.0
            v_max = 1.0
        self.hlg = HLGaussLoss(
            min_value=v_min,
            max_value=v_max,
            num_bins=self.rl_config.num_atoms,
            sigma=self.rl_config.sigma * ((v_max - v_min) / self.rl_config.num_atoms),
        )
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
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_value, config.tune_critic
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_value: bool, tune_critic: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_value = tune_value
        self.tune_critic = tune_critic
        for p in self.parameters():
            p.requires_grad = True
        self.target_critic.requires_grad_(False)
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            self.backbone_encoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_critic:
            self.ca_encoder.requires_grad_(False)
            self.critic.requires_grad_(False)
        if not tune_value:
            self.value.requires_grad_(False)
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
        critic_params = []

        # To avoid parameter overlap between groups, we will:
        # 1. Build a mapping from parameter to group name(s)
        # 2. Only assign each parameter to the first group it appears in (flow_matching > actor > critic)
        # 3. Remove duplicates

        # Collect all parameters for each group (with requires_grad)
        flow_matching_params = []
        value_params = []
        critic_params = []

        if self.tune_projector:
            flow_matching_params.extend(list(self.state_encoder.parameters()))
            flow_matching_params.extend(list(self.action_encoder.parameters()))
            flow_matching_params.extend(list(self.action_decoder.parameters()))
            if self.config.add_pos_embed:
                flow_matching_params.extend(list(self.position_embedding.parameters()))

        if self.tune_diffusion_model:
            flow_matching_params.extend(list(self.model.parameters()))

        if self.tune_value:
            value_params.extend(list(self.value.parameters()))

        if self.tune_critic:
            critic_params.extend(list(self.ca_encoder.parameters()))
            critic_params.extend(list(self.critic.parameters()))
            critic_params.extend(list(self.backbone_encoder.parameters()))

        # Only keep parameters that require gradients
        flow_matching_params = [p for p in flow_matching_params if p.requires_grad]
        critic_params = [p for p in critic_params if p.requires_grad]
        value_params = [p for p in value_params if p.requires_grad]

        return {
            "flow_matching": flow_matching_params,
            "value": value_params,
            "critic": critic_params,
        }
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
            if not self.tune_critic:
                self.critic.eval()
            if not self.tune_value:
                self.value.eval()
                self.backbone_encoder.eval()

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
        # NOTE: detach the vl_embeds to avoid gradient flow to the VLLN module in flow matching loss
        # Only flow gradient to the flow matching loss
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

    def compute_value_loss(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        # NOTE: detach the vl_embeds to avoid gradient flow to the VLLN module in value loss
        # Only value gradient to the value loss
        vl_embeds = backbone_output.backbone_features.detach()
        embodiment_id = action_input.embodiment_id

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        # Embed state.
        vl_embeds_mean = vl_embeds.mean(dim=1, keepdim=True)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
        vl_embed_features = F.tanh(vl_embed_features)

        v_logits = self.value(vl_embed_features, action_input.state)
        v_probs = torch.softmax(v_logits, dim=-1)
        vs = self.hlg.transform_from_probs(v_probs)

        # Value loss
        with torch.no_grad():
            vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
            vl_embed_features = F.tanh(vl_embed_features)
            q1_logits, q2_logits = self.target_critic(
                vl_embed_features, action_input.state, action_input.action[:, : self.critic_action_horizon]
            )
            q_logits = torch.stack([q1_logits, q2_logits], dim=0)
            q_probs = torch.softmax(q_logits, dim=-1)
            qs = self.hlg.transform_from_probs(q_probs)

            if self.rl_config.q_agg == "min":
                min_q_idx = torch.argmin(qs, dim=0)
                batch_indices = torch.arange(batch_size, device=device)
                q_logit = q_logits[min_q_idx, batch_indices]
                q_prob = torch.softmax(q_logit, dim=-1)
                q = self.hlg.transform_from_probs(q_prob)
            elif self.rl_config.q_agg == "mean":
                q_logit = q_logits / 2
                q_prob = q_probs / 2
                q = self.hlg.transform_from_probs(q_prob)
            else:
                assert False, f"Invalid q_agg: {self.rl_config.q_agg}"

        g_hard = torch.where(q >= vs, self.rl_config.expectile, 1 - self.rl_config.expectile)
        g_hard_ratio = torch.where(q >= vs, 1.0, 0.0).sum(dim=-1) / batch_size
        # Explicit cross entropy implementation: -sum(target * log_softmax(input))
        log_probs = F.log_softmax(v_logits, dim=-1)
        ce_loss = -(q_prob * log_probs).sum(dim=-1)
        value_loss = (g_hard * ce_loss).mean()

        metrics = {
            "target_q_mean": q.mean(),
            "target_q_std": q.std(),
            "target_q_min": q.min(),
            "target_q_max": q.max(),
            "v_mean": vs.mean(),
            "v_std": vs.std(),
            "v_min": vs.min(),
            "v_max": vs.max(),
            "expectile_ratio": g_hard_ratio.mean(),
        }
        return value_loss, metrics

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
        # NOTE: detach the vl_embeds to avoid gradient flow to the VLLN module in critic loss
        # Only critic gradient to the critic loss
        vl_embeds = backbone_output.backbone_features.detach()
        next_vl_embeds = next_backbone_output.backbone_features.detach()
        embodiment_id = action_input.embodiment_id

        vl_embeds_mean = vl_embeds.mean(dim=1, keepdim=True)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
        vl_embed_features = F.tanh(vl_embed_features)

        # Critic loss
        done = torch.prod(action_input.done, dim=-1)
        reward = action_input.reward
        if self.rl_config.negative_reward:
            reward -= 1
        discounts1 = self.rl_config.discount1 ** torch.arange(self.critic_action_horizon).to(reward.device)
        scaled_rewards = torch.sum(reward * discounts1, dim=-1)

        with torch.no_grad():
            next_vl_embeds_mean = next_vl_embeds.mean(dim=1, keepdim=True)
            next_vl_embed_features = self.backbone_encoder(next_vl_embeds_mean, embodiment_id)
            next_vl_embed_features = F.tanh(next_vl_embed_features)

            v_logits = self.value(next_vl_embed_features, action_input.next_state)
            v_probs = torch.softmax(v_logits, dim=-1)
            vs = self.hlg.transform_from_probs(v_probs)

            target_v = (
                scaled_rewards
                + (self.rl_config.discount2 ** (self.rl_config.nstep * self.critic_action_horizon)) * (1.0 - done) * vs
            )

        q1_logits, q2_logits = self.critic(
            vl_embed_features, action_input.state, action_input.action[:, : self.critic_action_horizon]
        )

        q1_probs = torch.softmax(q1_logits, dim=-1)
        q2_probs = torch.softmax(q2_logits, dim=-1)
        q1 = self.hlg.transform_from_probs(q1_probs)
        q2 = self.hlg.transform_from_probs(q2_probs)

        critic_loss = (self.hlg(q1_logits, target_v) + self.hlg(q2_logits, target_v)) / 2

        metrics = {
            "target_v_mean": target_v.mean(),
            "target_v_std": target_v.std(),
            "target_v_min": target_v.min(),
            "target_v_max": target_v.max(),
            "q1_mean": q1.mean(),
            "q1_std": q1.std(),
            "q1_min": q1.min(),
            "q1_max": q1.max(),
            "q2_mean": q2.mean(),
            "q2_std": q2.std(),
            "q2_min": q2.min(),
            "q2_max": q2.max(),
            "batch_reward": scaled_rewards.detach().mean(),
        }
        return critic_loss, metrics

    def forward(
        self, backbone_output: BatchFeature, next_backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        # Compute each loss separately to avoid gradient conflicts
        flow_matching_loss = self.compute_flow_matching_loss(backbone_output, action_input)
        value_loss, value_metrics = self.compute_value_loss(backbone_output, action_input)
        critic_loss, critic_metrics = self.compute_critic_loss(backbone_output, next_backbone_output, action_input)
        total_loss = flow_matching_loss + value_loss + critic_loss

        output_dict = {
            "loss": total_loss,
            "flow_matching_loss": flow_matching_loss,
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            **{f"value/{k}": v for k, v in value_metrics.items()},
            **{f"critic/{k}": v for k, v in critic_metrics.items()},
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(self.rl_config.num_samples * batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # repeat state_features and embodiment_id for num_samples times
        state_features = state_features.repeat(self.rl_config.num_samples, 1, 1)
        embodiment_id = embodiment_id.repeat(self.rl_config.num_samples)
        vl_embs = vl_embs.repeat(self.rl_config.num_samples, 1, 1)

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(self.rl_config.num_samples * batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        vl_embeds_mean = vl_embs.mean(dim=1, keepdim=True)
        vl_embed_features = self.backbone_encoder(vl_embeds_mean, embodiment_id)
        vl_embed_features = F.tanh(vl_embed_features)

        state = action_input.state.repeat(self.rl_config.num_samples, 1, 1)
        q1_logits, q2_logits = self.critic(vl_embed_features, state, actions[:, : self.critic_action_horizon])
        q1_probs, q2_probs = torch.softmax(q1_logits, dim=-1), torch.softmax(q2_logits, dim=-1)
        q1, q2 = self.hlg.transform_from_probs(q1_probs), self.hlg.transform_from_probs(q2_probs)
        q = torch.min(q1, q2)

        q = q.reshape(self.rl_config.num_samples, batch_size)
        actions = actions.reshape(
            self.rl_config.num_samples, batch_size, self.config.action_horizon, self.config.action_dim
        )

        # Select actions with highest q values
        # (batch_size,)
        if self.rl_config.temperature > 0:
            q_dists = F.softmax(q / self.rl_config.temperature, dim=0)
            # Randomly sample indices according to q_dists (softmaxed q values)
            # q_dists: (num_samples, batch_size)
            # For each batch, sample one index from num_samples according to q_dists[:, i]
            # Use torch.distributions.Categorical for sampling indices
            cat_dist = torch.distributions.Categorical(probs=q_dists.transpose(0, 1))
            selected_indices = cat_dist.sample()
        else:
            selected_indices = torch.argmax(q, dim=0)
        # (batch_size, action_horizon, action_dim)
        selected_actions = actions[selected_indices, torch.arange(batch_size)]

        # Apply critic action horizon if needed
        if hasattr(self, "critic_action_horizon") and self.critic_action_horizon < self.config.action_horizon:
            selected_actions = selected_actions[:, : self.critic_action_horizon]

        return BatchFeature(data={"action_pred": selected_actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
