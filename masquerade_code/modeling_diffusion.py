"""
Masquerade version of lerobot/policies/diffusion/modeling_diffusion.py
"""

#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""
import pdb
import math
from collections import deque
from typing import Callable
import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.constants import OBS_ENV, OBS_ROBOT, OBS_LANGUAGE_EMBEDDING
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize, NormalizeMultiDataset, UnnormalizeMultiDataset
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.common.policies.diffusion.visualization_utils import debug_crop_visualization, debug_aux_loss_visualization


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        if config.use_auxiliary_mlp:
            self.normalize_inputs = NormalizeMultiDataset(config.input_features, config.normalization_mapping, dataset_stats)
            self.normalize_targets = NormalizeMultiDataset(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_inputs = UnnormalizeMultiDataset(
                config.input_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = UnnormalizeMultiDataset(
                config.output_features, config.normalization_mapping, dataset_stats
            )
        else:
            print("Using Normalize")
            print("Dataset stats: ", dataset_stats)
            self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
            self.normalize_targets = Normalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_inputs = Unnormalize(
                config.input_features, config.normalization_mapping, dataset_stats
            )
            self.unnormalize_outputs = Unnormalize(
                config.output_features, config.normalization_mapping, dataset_stats
            )
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config, self.unnormalize_inputs, self.unnormalize_outputs)

        self.reset()

    def get_optim_params(self) -> dict:
        if not self.config.freeze_rgb_encoders:
            return self.diffusion.parameters()
        else:
            return self.get_optim_params_excluding_frozen_rgb()

    def get_optim_params_excluding_frozen_rgb(self) -> dict:
        """Get parameters for optimization, excluding frozen RGB encoder parameters.
        
        This method filters out parameters from frozen RGB encoders to potentially
        reduce memory usage and computation in the optimizer.
        
        When finetune_layernorm_rgb_encoders is True, LayerNorm parameters from RGB encoders
        will be included in the returned parameters.
        
        Returns:
            Parameters that should be optimized (excluding frozen RGB encoder params).
        """
        # Filter out frozen RGB encoder parameters
        trainable_params = []
        
        # Add all parameters except those from frozen RGB encoders
        for name, param in self.diffusion.named_parameters():
            # Skip parameters from RGB encoders if they're frozen
            if name.startswith('rgb_encoder') and not param.requires_grad:
                continue
            trainable_params.append(param)
        
        return trainable_params

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
            "observation.language_embedding": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        if self.config.use_auxiliary_mlp:
            # mask = batch["dataset_index"]
            mask = torch.zeros((batch["observation.state"].shape[0]))
            batch = self.normalize_inputs(batch, mask)
        else:
            batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            if self.config.use_auxiliary_mlp:
                # mask = batch["dataset_index"]
                mask = torch.zeros((batch["observation.state"].shape[0])).int()
                actions = self.unnormalize_outputs({"action": actions}, mask)["action"]
            else:
                actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor], training_step: int | None = None) -> tuple[Tensor, dict, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.use_auxiliary_mlp:
            mask = batch["dataset_index"]
            batch = self.normalize_inputs(batch, mask)
        else:
            batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        if self.config.use_auxiliary_mlp:
            mask = batch["dataset_index"]
            batch = self.normalize_targets(batch, mask)
        else:
            batch = self.normalize_targets(batch)
        loss, auxiliary_output_dict = self.diffusion.compute_loss(batch, training_step)
        output_dict = None
        return loss, output_dict, auxiliary_output_dict

def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig, unnormalize_inputs: Callable, unnormalize_outputs: Callable):
        super().__init__()
        self.config = config
        self.unnormalize_inputs = unnormalize_inputs
        self.unnormalize_outputs = unnormalize_outputs

        # Build observation encoders (depending on which observations are provided).
        if self.config.use_proprioception:
            global_cond_dim = self.config.robot_state_feature.shape[0]
        else:
            global_cond_dim = 0
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
        if not self.config.no_diffusion_loss:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        else:
            self.unet = None

        # Add auxiliary MLP branch if enabled
        if (config.use_auxiliary_mlp or config.no_diffusion_loss) and config.image_features:
            # Determine the RGB feature dimension
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_feature_dim = encoders[0].feature_dim * len(config.image_features)
            else:
                self.rgb_feature_dim = self.rgb_encoder.feature_dim * len(config.image_features)
            
            # Build auxiliary MLP
            feature_dim = 50  # Intermediate feature dimension
            bc_hidden_dim = 512  # Hidden dimension for BC policy

            # Feature trunk: projects visual features to intermediate representation
            self.auxiliary_trunk = nn.Sequential(
                nn.Linear(self.rgb_feature_dim, feature_dim),
                nn.LayerNorm(feature_dim), 
                nn.Tanh()
            )

            # Policy head: predicts actions from features
            self.auxiliary_policy = nn.Sequential(
                nn.Linear(feature_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, config.auxiliary_mlp_output_dim)
            )

            if config.use_film_cond:
                film_hidden_dim = config.film_hidden_dim
                self.film_network = nn.Sequential(
                    nn.Linear(768, film_hidden_dim),  # 768 is typical BERT embedding size
                    nn.ReLU(),
                    nn.Linear(film_hidden_dim, film_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(film_hidden_dim, 2 * self.rgb_feature_dim)  # 2 * outdim for scale and bias
                )
            auxiliary_trunk_weights = {}
            auxiliary_policy_weights = {}
            film_weights = {}
            if config.pretrained_backbone_weights and "snapshot" in config.pretrained_backbone_weights:
                weights = torch.load(config.pretrained_backbone_weights, weights_only=True)['visual_encoder']
                for k, v in weights.items():
                    if k.startswith('module.bc_trunk'):
                        auxiliary_trunk_weights[k.replace('module.bc_trunk.', '')] = v
                    if k.startswith('module.bc_policy'):
                        if config.no_diffusion_loss:
                            if (not "4.weight" in k) and (not "4.bias" in k):
                                auxiliary_policy_weights[k.replace('module.bc_policy.', '')] = v
                        else:
                            auxiliary_policy_weights[k.replace('module.bc_policy.', '')] = v
                    if config.use_film_cond:
                        if k.startswith('module.film_network'):
                            film_weights[k.replace('module.film_network.', '')] = v                        
                self.auxiliary_trunk.load_state_dict(auxiliary_trunk_weights)
                self.auxiliary_policy.load_state_dict(auxiliary_policy_weights, strict=True)
                if config.use_film_cond:
                    self.film_network.load_state_dict(film_weights)
        else:
            self.auxiliary_trunk = None
            self.auxiliary_policy = None
            self.film_network = None

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

        # Freeze RGB encoders if specified in config
        if config.freeze_rgb_encoders:
            self._freeze_rgb_encoders()

    def _freeze_rgb_encoders(self):
        """Freeze the RGB encoder parameters to prevent them from being updated during training.
        
        If finetune_layernorm_rgb_encoders is True, only LayerNorm parameters will remain trainable.
        """
        if hasattr(self, 'rgb_encoder'):
            if isinstance(self.rgb_encoder, nn.ModuleList):
                # Multiple encoders (one per camera)
                for encoder in self.rgb_encoder:
                    self._freeze_encoder_parameters(encoder)
            else:
                # Single encoder
                self._freeze_encoder_parameters(self.rgb_encoder)

    def _freeze_encoder_parameters(self, encoder):
        """Freeze parameters in a single encoder, with optional LayerNorm finetuning."""
        for name, param in encoder.named_parameters():
            # If LayerNorm finetuning is enabled, keep LayerNorm parameters trainable
            if self.config.finetune_layernorm_rgb_encoders and self._is_layernorm_parameter(name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _is_layernorm_parameter(self, param_name: str) -> bool:
        """Check if a parameter belongs to a LayerNorm layer.
        
        Args:
            param_name: The name of the parameter
            
        Returns:
            True if the parameter is from a LayerNorm layer, False otherwise
        """
        # LayerNorm parameters typically end with '.weight' or '.bias'
        # and are found in modules named 'norm', 'ln', 'layer_norm', etc.
        layernorm_indicators = ['norm', 'ln', 'layer_norm', 'layernorm']
        return any(indicator in param_name.lower() for indicator in layernorm_indicators)

    def unfreeze_rgb_encoders(self):
        """Unfreeze the RGB encoder parameters to allow them to be updated during training."""
        if hasattr(self, 'rgb_encoder'):
            if isinstance(self.rgb_encoder, nn.ModuleList):
                # Multiple encoders (one per camera)
                for encoder in self.rgb_encoder:
                    for param in encoder.parameters():
                        param.requires_grad = True
            else:
                # Single encoder
                for param in self.rgb_encoder.parameters():
                    param.requires_grad = True

    def print_parameter_status(self):
        """Print the status of all parameters in RGB encoders for debugging."""
        if not hasattr(self, 'rgb_encoder'):
            print("No RGB encoder found.")
            return
            
        print("RGB Encoder Parameter Status:")
        print("=" * 50)
        
        if isinstance(self.rgb_encoder, nn.ModuleList):
            # Multiple encoders (one per camera)
            for i, encoder in enumerate(self.rgb_encoder):
                print(f"\nCamera {i} Encoder:")
                self._print_encoder_parameter_status(encoder)
        else:
            # Single encoder
            print("\nSingle Encoder:")
            self._print_encoder_parameter_status(self.rgb_encoder)

    def _print_encoder_parameter_status(self, encoder):
        """Print parameter status for a single encoder."""
        total_params = 0
        trainable_params = 0
        layernorm_params = 0
        trainable_layernorm_params = 0
        
        for name, param in encoder.named_parameters():
            total_params += 1
            is_layernorm = self._is_layernorm_parameter(name)
            is_trainable = param.requires_grad
            
            if is_trainable:
                trainable_params += 1
            if is_layernorm:
                layernorm_params += 1
                if is_trainable:
                    trainable_layernorm_params += 1
                    
            status = "TRAINABLE" if is_trainable else "FROZEN"
            layernorm_indicator = " [LayerNorm]" if is_layernorm else ""
            print(f"  {name}: {status}{layernorm_indicator}")
        
        print(f"\n  Summary: {trainable_params}/{total_params} parameters trainable")
        print(f"  LayerNorm: {trainable_layernorm_params}/{layernorm_params} parameters trainable")

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _extract_rgb_features(self, batch: "dict[str, Tensor]", training_step: int | None = None) -> tuple[Tensor, Tensor]:
        """Extract RGB features from images for auxiliary tasks."""
        if not self.config.image_features:
            return None, None
            
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        if self.config.use_auxiliary_mlp:
            random_crop_mask = torch.ones((batch["observation.state"].shape[0]))
            # Set random_crop_mask to int type
            random_crop_mask = random_crop_mask.int()
            # random_crop_mask = (batch["dataset_index"] != 1)
        else:
            random_crop_mask = None
        
        # Expand random_crop_mask to match reshaped images: (b,) -> (b s,) for each camera
        if random_crop_mask is not None:
            # Expand to (b, s) then flatten to (b s)
            num_cameras = batch["observation.images"].shape[2]
            expanded_mask = random_crop_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, n_obs_steps, num_cameras).flatten()
        else:
            expanded_mask = None
        
        if self.config.use_separate_rgb_encoder_per_camera:
            # Combine batch and sequence dims while rearranging to make the camera index dimension first.
            images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
            
            img_features_list = []
            img_feataures_mlp_list = []
            for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True):
                img_features, img_features_mlp = encoder(images, random_crop_mask=expanded_mask, training_step=training_step)
                img_features_list.append(img_features)
                img_feataures_mlp_list.append(img_features_mlp)
            img_features_list = torch.cat(img_features_list, dim=0)
            img_feataures_mlp_list = torch.cat(img_feataures_mlp_list, dim=0)
            # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            img_features_mlp = einops.rearrange(
                img_feataures_mlp_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
            img_features, img_features_mlp = self.rgb_encoder(
                einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ..."),
                random_crop_mask=expanded_mask,
                training_step=training_step
            )
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            img_features_mlp = einops.rearrange(
                img_features_mlp, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        
        return img_features, img_features_mlp

    def _prepare_global_conditioning(self, batch: "dict[str, Tensor]", training_step: int | None = None) -> tuple[Tensor, Tensor]:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        if self.config.use_proprioception:
            global_cond_feats = [batch[OBS_ROBOT]]
        else:
            global_cond_feats = []
        # Extract image features.
        img_features, img_features_mlp = self._extract_rgb_features(batch, training_step)
        if self.config.image_features:
            global_cond_feats.append(img_features)
            # global_cond_feats.append(img_features_mlp)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1), img_features_mlp

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, rgb_features_mlp = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        if not self.config.no_diffusion_loss:
            actions = self.conditional_sample(batch_size, global_cond=global_cond) # diffusion
        else:
            # mlp
            rgb_features_mlp = rgb_features_mlp[:, 1, :]
            if self.config.use_film_cond and batch["observation.language_embedding"] is not None:
                rgb_features_mlp = self._apply_film_conditioning(batch, rgb_features_mlp)
            actions = self.auxiliary_policy(self.auxiliary_trunk(rgb_features_mlp)) 
            actions = actions.reshape(-1, 8, 20)
    
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor], training_step: int | None = None) -> tuple[Tensor, dict]:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_step import timms, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        
        Args:
            batch: Input batch dictionary
            training_step: Current training step for debug logging
            
        Returns:
            Tensor (total loss)
            dict (output_dict)
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        batch_size = batch["action"].shape[0]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, rgb_features_mlp = self._prepare_global_conditioning(batch, training_step)  # (B, global_cond_dim)
        if not self.config.no_diffusion_loss:
            # Compute auxiliary loss if enabled
            aux_loss, aux_loss_per_sample, aux_pred_for_viz, aux_loss_left_per_sample, aux_loss_right_per_sample = self._compute_auxiliary_loss(
                batch, rgb_features_mlp, batch_size, horizon
            )

            # Store auxiliary loss values and handle debug visualization
            self._handle_auxiliary_loss_debug(
                batch, aux_loss_per_sample, aux_loss_left_per_sample, aux_loss_right_per_sample,
                aux_pred_for_viz, training_step
            )

        if self.config.no_diffusion_loss:
            # Extract RGB features for auxiliary task
            rgb_features_mlp = rgb_features_mlp[:, 1, :]
            
            # Apply FiLM conditioning if enabled and language embeddings provided
            if self.config.use_film_cond and batch["observation.language_embedding"] is not None:
                rgb_features_mlp = self._apply_film_conditioning(batch, rgb_features_mlp)
            
            # Generate auxiliary predictions
            aux_pred = self.auxiliary_policy(self.auxiliary_trunk(rgb_features_mlp))
            horizon_offset = int(horizon)
            aux_pred = aux_pred.reshape(batch_size, horizon_offset, -1)
            aux_loss = F.mse_loss(aux_pred, batch["action"][:, :horizon_offset, :], reduction="none")
            aux_loss = aux_loss.mean()
            return aux_loss, {"weighted_aux_loss": aux_loss}

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            diffusion_loss = diffusion_loss * in_episode_bound.unsqueeze(-1)
        
        # Mask loss for 2D action if enabled
        if self.config.do_mask_loss_for_2d_action:
            diffusion_loss = self._apply_2d_action_mask(batch, diffusion_loss, batch_size, horizon)
        if self.config.use_auxiliary_mlp:
            batch_dataset_mask = (batch["dataset_index"] != 1)
            batch_dataset_mask = batch_dataset_mask.unsqueeze(-1).expand(batch_size, horizon)
            diffusion_loss = diffusion_loss * batch_dataset_mask.unsqueeze(-1)
            total_valid_elements = torch.sum(diffusion_loss > 0)
            # print(f"diffusion_loss: {diffusion_loss.shape}")
            # print(f"total_valid_elements diffusion_loss: {total_valid_elements}")
            diffusion_loss = diffusion_loss.sum() / (total_valid_elements.float()+1e-8)
        else:
            diffusion_loss = diffusion_loss.mean()

        # Combine losses and return
        return self._combine_losses(aux_loss, diffusion_loss) 
    
    def _compute_auxiliary_loss(
        self, batch: dict[str, Tensor], rgb_features_mlp: Tensor, batch_size: int, horizon: int
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Compute auxiliary loss for auxiliary MLP if enabled."""
        if not (self.config.use_auxiliary_mlp and self.auxiliary_policy is not None):
            return None, None, None, None, None
        
        if rgb_features_mlp is None:
            return None, None, None, None, None
        
        # Extract RGB features for auxiliary task
        rgb_features_mlp = rgb_features_mlp[:, 1, :]
        
        # Apply FiLM conditioning if enabled and language embeddings provided
        if self.config.use_film_cond and batch["observation.language_embedding"] is not None:
            rgb_features_mlp = self._apply_film_conditioning(batch, rgb_features_mlp)
        
        # Generate auxiliary predictions
        aux_pred = self.auxiliary_policy(self.auxiliary_trunk(rgb_features_mlp))
        horizon_offset = int(horizon / 2)
        aux_pred = aux_pred.reshape(batch_size, horizon_offset, -1)
        
        # Left hand 2D pixel action: 0-1, Right hand 2D pixel action: 10-11
        if self.config.auxiliary_mlp_output_dim == 32:
            auxiliary_targets = batch["action"][:, :horizon_offset, [0,1,10,11]]
        elif self.config.auxiliary_mlp_output_dim == 144:
            target_indices = list(range(0, 20))
            target_indices.remove(2)
            target_indices.remove(12)
            auxiliary_targets = batch["action"][:, :horizon_offset, target_indices]
        
        # Store for visualization
        aux_pred_for_viz = aux_pred.detach()
        
        # Compute masked auxiliary losses
        aux_loss, aux_loss_left_per_sample, aux_loss_right_per_sample = self._compute_masked_auxiliary_loss(
            batch, aux_pred, auxiliary_targets, batch_size, horizon_offset
        )
        
        # Compute per-sample auxiliary losses for visualization
        aux_loss_per_sample = aux_loss.mean(dim=(1, 2))
        total_valid_elements = torch.sum(aux_loss > 0)
        # print(f"aux_loss: {aux_loss.shape}")
        # print(f"total_valid_elements: {total_valid_elements}")
        aux_loss = aux_loss.sum() / (total_valid_elements.float()+1e-8)
        
        return aux_loss, aux_loss_per_sample, aux_pred_for_viz, aux_loss_left_per_sample, aux_loss_right_per_sample
    
    def _apply_film_conditioning(self, batch: dict[str, Tensor], rgb_features_mlp: Tensor) -> Tensor:
        """Apply FiLM conditioning to RGB features."""
        # Generate FiLM parameters from language embedding
        cond_embed = self.film_network(batch["observation.language_embedding"][:, 1, :])
        
        # Ensure cond_embed has the right shape: [batch_size, 2 * outdim]
        # First half is scale, second half is bias
        if cond_embed.shape[-1] != 2 * self.rgb_feature_dim:
            raise ValueError(f"cond_embed should have shape [..., {2 * self.rgb_feature_dim}], got {cond_embed.shape}")
        
        # Split FiLM parameters into scale and bias
        scale = cond_embed[..., :self.rgb_feature_dim]
        bias = cond_embed[..., self.rgb_feature_dim:]

        # Apply FiLM conditioning: h = scale * h + bias
        return scale * rgb_features_mlp + bias
    
    def _compute_masked_auxiliary_loss(
        self, batch: dict[str, Tensor], aux_pred: Tensor, auxiliary_targets: Tensor, 
        batch_size: int, horizon_offset: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute masked auxiliary loss for left and right hands."""
        # Get action mask for left and right hand in frame
        action_left_in_frame = (~batch["action_2d_left_is_out_of_frame"]).expand((batch_size, horizon_offset))
        action_right_in_frame = (~batch["action_2d_right_is_out_of_frame"]).expand((batch_size, horizon_offset))

        # Left aux loss and Right aux loss
        aux_loss_left = F.mse_loss(aux_pred[:, :, :2], auxiliary_targets[:, :, :2], reduction="none")
        aux_loss_right = F.mse_loss(aux_pred[:, :, 2:4], auxiliary_targets[:, :, 2:4], reduction="none")

        # Mask loss for left and right hand out of frame
        aux_loss_left_masked = aux_loss_left * action_left_in_frame.unsqueeze(-1)
        aux_loss_right_masked = aux_loss_right * action_right_in_frame.unsqueeze(-1)
        aux_loss = aux_loss_left_masked + aux_loss_right_masked

        # Don't mask out aux loss if the dataset is epic, otherwise mask out the aux loss
        # FIXME: Epic dataset is indexed as 1, this is hardcoded for now
        batch_dataset_mask = (batch["dataset_index"] == 1)
        batch_dataset_mask = batch_dataset_mask.unsqueeze(-1).expand(batch_size, horizon_offset)
        aux_loss = aux_loss * batch_dataset_mask.unsqueeze(-1)
        
        # Compute per-sample auxiliary losses for visualization
        aux_loss_left_per_sample = (aux_loss_left_masked * batch_dataset_mask.unsqueeze(-1)).mean(dim=(1, 2))
        aux_loss_right_per_sample = (aux_loss_right_masked * batch_dataset_mask.unsqueeze(-1)).mean(dim=(1, 2))
        
        return aux_loss, aux_loss_left_per_sample, aux_loss_right_per_sample
    
    def _handle_auxiliary_loss_debug(
        self, batch: dict[str, Tensor], aux_loss_per_sample: Tensor | None, 
        aux_loss_left_per_sample: Tensor | None, aux_loss_right_per_sample: Tensor | None,
        aux_pred_for_viz: Tensor | None, training_step: int | None
    ) -> None:
        """Store auxiliary loss values and handle debug visualization."""
        # Store auxiliary loss values for debug visualization
        if aux_loss_per_sample is not None:
            self._current_aux_loss_per_sample = aux_loss_per_sample.detach()
            self._current_aux_loss_left_per_sample = aux_loss_left_per_sample.detach()
            self._current_aux_loss_right_per_sample = aux_loss_right_per_sample.detach()
        else:
            self._current_aux_loss_per_sample = None
            self._current_aux_loss_left_per_sample = None
            self._current_aux_loss_right_per_sample = None

        # Debug visualization for auxiliary loss every k training steps
        if (self.config.cotrain_debug and training_step is not None and aux_loss_per_sample is not None 
            and training_step % self.config.cotrain_debug_freq == 0):
            debug_aux_loss_visualization(
                batch, aux_loss_per_sample, training_step, 
                self.unnormalize_inputs, self.unnormalize_outputs,
                aux_pred_for_viz, aux_loss_left_per_sample, 
                aux_loss_right_per_sample
            )
    
    def _apply_2d_action_mask(self, batch: dict[str, Tensor], diffusion_loss: Tensor, batch_size: int, horizon: int) -> Tensor:
        """Apply 2D action mask to diffusion loss."""
        # Init action mask
        action_mask = torch.ones_like(diffusion_loss)

        # Get action mask for left and right hand in frame
        action_left_in_frame = (~batch["action_2d_left_is_out_of_frame"]).expand((batch_size, horizon))
        action_right_in_frame = (~batch["action_2d_right_is_out_of_frame"]).expand((batch_size, horizon))

        # Mask loss for left and right hand in frame
        action_mask[:, :, :10] = action_mask[:, :, :10] * action_left_in_frame.unsqueeze(-1)
        action_mask[:, :, 10:20] = action_mask[:, :, 10:20] * action_right_in_frame.unsqueeze(-1)

        # Mask loss for 2D epic loss (Don't mask anything if the dataset is not epic)
        # FIXME: Epic dataset is indexed as 1, this is hardcoded for now
        batch_dataset_mask = (batch["dataset_index"] != 1).unsqueeze(-1).expand(batch_size, horizon)

        # If the dataset is epic, mask out rotation 6d and gripper action, which are 2-10 for left hand and 12-20 for right hand
        action_mask[:, :, 2:10] = action_mask[:, :, 2:10] * batch_dataset_mask.unsqueeze(-1)
        action_mask[:, :, 12:20] = action_mask[:, :, 12:20] * batch_dataset_mask.unsqueeze(-1)
        
        # Debug output
        print(action_mask[:10, 0])
        print(f"action_mask: {action_mask.shape}")
        
        return diffusion_loss * action_mask
    
    def _combine_losses(self, aux_loss: Tensor | None, diffusion_loss: Tensor) -> tuple[Tensor, dict | None]:
        """Combine auxiliary and diffusion losses and return appropriate structure."""
        if aux_loss is not None:
            total_loss = diffusion_loss + self.config.auxiliary_loss_weight * aux_loss
            return total_loss, {"weighted_aux_loss": self.config.auxiliary_loss_weight * aux_loss, "diffusion_loss": diffusion_loss}
        else:
            return diffusion_loss, None



class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        self.config = config
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone (example, change path for specific training run)
        config.pretrained_backbone_weights = "/home/masquerade/outputs/v20/snapshot.pt"

        if config.pretrained_backbone_weights and "snapshot" in config.pretrained_backbone_weights:
            weights = torch.load(config.pretrained_backbone_weights, weights_only=True)['visual_encoder']
            new_weights = {}
            for key, value in weights.items():
                if 'module.encoder' in key:
                    new_weights[key.replace('module.encoder.', '')] = value
            if "resnet" in config.vision_backbone:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=None)
            elif "dinov2" in config.vision_backbone:
                backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            elif "vit" in config.vision_backbone:
                import timm
                backbone_model = timm.create_model('vit_base_patch16_224.mae')
            backbone_model.load_state_dict(new_weights, strict=True)
            print(f"Loaded pretrained weights for {config.pretrained_backbone_weights}")
        else:
            # Check if it's a timm model (e.g., vit_mae)
            if "vit" in config.vision_backbone:
                import timm
                if config.pretrained_backbone_weights is not None:
                    if "dinov2" not in config.pretrained_backbone_weights and "hrp" not in config.pretrained_backbone_weights:
                        print("LOADING TIMM BACKBONE", config.pretrained_backbone_weights)
                        backbone_model = timm.create_model(config.pretrained_backbone_weights, pretrained=True)
                        # print(f"Loaded pretrained weights for {config.pretrained_backbone_weights}")
                    elif "dinov2" in config.pretrained_backbone_weights:
                        backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                        # print(f"Loaded pretrained weights for {config.pretrained_backbone_weights}")
                    elif "hrp" in config.pretrained_backbone_weights:
                        backbone_model = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
                        weights = torch.load(config.pretrained_backbone_weights, weights_only=True)['model']
                        backbone_model.load_state_dict(weights, strict=True)
                        print(f"Loaded pretrained weights for {config.pretrained_backbone_weights}")
                    else:
                        raise ValueError(f"Unsupported pretrained backbone weights {config.pretrained_backbone_weights}")
                else:
                    backbone_model = timm.create_model('vit_base_patch16_224.mae', pretrained=False)
            elif "dinov2" in config.vision_backbone:
                print("LOADING DINOV2 BACKBONE")
                backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                print(f"Loaded pretrained weights for {config.pretrained_backbone_weights}")
            else:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    weights=config.pretrained_backbone_weights
                )


        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        if "vit" in config.vision_backbone or "dinov2" in config.vision_backbone:
            # For ViT models, we'll use the entire model as backbone
            self.backbone = backbone_model
        else:
            # For CNN models, use the feature extractor (remove classifier)
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        
        if "vit" in config.vision_backbone or "dinov2" in config.vision_backbone:
            # For ViT models, get feature dimension from the model
            with torch.no_grad():
                dummy_input = torch.randn(dummy_shape)
                # if config.pretrained_backbone_weights == "vit_base_224_dinov2":
                #     features = backbone_model(self.dinov2_processor(dummy_input, return_tensors="pt").pixel_values)
                # else:
                features = backbone_model(dummy_input)
                feature_dim = features.shape[-1]
            self.pool = None  # No pooling needed for ViT

        else:
            # For CNN models, use SpatialSoftmax
            feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
            self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
            feature_dim = config.spatial_softmax_num_keypoints * 2

        self.feature_dim = feature_dim
        self.out = nn.Linear(feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, random_crop_mask: Tensor | None = None, training_step: int | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
            random_crop_mask: (B,) boolean tensor indicating which samples need to be randomly cropped, or None for default cropping
            training_step: Current training step for debug logging (optional)
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if random_crop_mask is not None:
                # Apply cropping selectively based on crop_mask - vectorized version
                crop_indices = torch.where(random_crop_mask)[0]
                no_crop_indices = torch.where(~random_crop_mask)[0]
                
                # Create output tensor
                x_out = torch.zeros((x.shape[0], x.shape[1], self.config.crop_shape[0], self.config.crop_shape[1])).to(x.device)
                
                if len(crop_indices) > 0:
                    # Extract images that need random/center cropping and apply cropping in batch
                    images_to_crop = x[crop_indices]
                    if self.training:  # noqa: SIM108
                        cropped_images = self.maybe_random_crop(images_to_crop)
                    else:
                        # Always use center crop for eval.
                        cropped_images = self.center_crop(images_to_crop)
                    x_out[crop_indices] = cropped_images
                
                if len(no_crop_indices) > 0:
                    # For mask==0 samples, always do center crop
                    images_to_center_crop = x[no_crop_indices]
                    center_cropped_images = self.center_crop(images_to_center_crop)
                    x_out[no_crop_indices] = center_cropped_images
                
                x = x_out
                # # Debug visualization every k training steps
                # if self.config.cotrain_debug and random_crop_mask is not None and training_step is not None:
                #     if training_step % self.config.cotrain_debug_freq == 0:  # Every k steps
                #         debug_crop_visualization(x, random_crop_mask, training_step, self.training)
            else:
                # Fallback to original cropping behavior - apply to all samples
                if self.training:  # noqa: SIM108
                    x = self.maybe_random_crop(x)
                else:
                    # Always use center crop for eval.
                    x = self.center_crop(x)
        
        # Extract backbone feature.
        if "vit" in self.config.vision_backbone or "dinov2" in self.config.vision_backbone:
            # For ViT models, backbone outputs 1D features directly
            img_features = self.backbone(x)
            # ViT outputs (B, feature_dim), so we don't need flattening
        else:
            # For CNN models, use SpatialSoftmax pooling
            img_features = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        
        # Final linear layer with non-linearity.
        x = self.relu(self.out(img_features))
        return x, img_features


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out