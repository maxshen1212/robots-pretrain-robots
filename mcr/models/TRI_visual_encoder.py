"""
Visual Encoder (VisualEncoder) Model

This module implements the VisualEncoder model for learning visual representations from large-scale robot datasets.
The model supports multiple loss functions, various visual encoders, and language conditioning through FiLM.

Key Features:
- Multiple visual encoders (ResNet, ViT, DINO)
- FiLM conditioning for language-guided representations
- Behavioral cloning for action prediction
- Temporal contrastive learning
- Video-language alignment

Code based on https://github.com/luccachiang/robots-pretrain-robots/tree/main
"""

import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from visual_encoder import utils
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.nn.functional as F

# Small epsilon value to prevent division by zero
epsilon = 1e-8


def do_nothing(x):
    """Identity function that returns input unchanged."""
    return x


class VisualEncoder(nn.Module):
    """
    Visual Encoder (VisualEncoder) Model

    This is the main model class that combines visual encoding, language conditioning,
    and multiple loss functions for robot representation learning.

    The model architecture consists of:
    1. Visual encoder (ResNet/ViT/DINO/CLIP)
    2. FiLM conditioning network (optional)
    3. Language reward network (optional)
    4. State encoder (optional)
    5. Behavioral cloning policy (optional)
    """

    def __init__(
        self,
        device,
        lr,
        model_name="resnet50",
        l2dist=True,
        bc_weight=1.0,
        l2weight=0.0,
        l1weight=0.0,
        pretrained=False,
        use_film_cond=False,
        layernorm_finetune=False,
    ):
        """
        Initialize the VisualEncoder model.

        Args:
            device (str): Device to run model on ("cuda" or "cpu")
            lr (float): Learning rate for optimizer
            model_name (str): Visual encoder architecture ("resnet18", "resnet34", "resnet50", "vit", etc.)
            l2dist (bool): Whether to use L2 distance (True) or cosine similarity (False)
            bc_weight (float): Weight for behavioral cloning loss
            l2weight (float): Weight for L2 regularization loss
            l1weight (float): Weight for L1 regularization loss
            pretrained (bool): Whether to use pretrained visual encoder
            use_film_cond (bool): Whether to enable FiLM conditioning
            layernorm_finetune (bool): Whether to freeze all parameters except LayerNorm parameters
        """
        super().__init__()

        # Store device and basic settings
        self.device = device
        self.use_tb = False  # TensorBoard logging flag
        self.l2dist = (
            l2dist  # Use L2 distance (True) or cosine similarity (False)
        )
        self.pretrained = pretrained  # Whether to use pretrained visual encoder
        self.num_negatives = (
            3  # Number of negative samples for contrastive learning
        )
        self.use_film_cond = use_film_cond  # Enable FiLM conditioning
        self.layernorm_finetune = (
            layernorm_finetune  # Enable LayerNorm fine-tuning
        )

        # Loss weights
        self.bc_weight = bc_weight  # Weight on behavioral cloning loss
        self.l2weight = l2weight  # Weight on L2 regularization loss
        self.l1weight = l1weight  # Weight on L1 regularization loss

        # Initialize distance metrics and loss functions
        self.cs = torch.nn.CosineSimilarity(
            1
        )  # Cosine similarity for contrastive learning
        self.bce = nn.BCELoss(reduce=False)  # Binary cross-entropy loss
        self.sigm = Sigmoid()  # Sigmoid activation
        self.bc_loss = nn.MSELoss(
            reduction="none"
        )  # MSE loss for behavioral cloning

        print("pretrained", self.pretrained)
        if self.layernorm_finetune:
            print(
                "LayerNorm fine-tuning enabled - freezing all parameters except LayerNorm"
            )

        # List to collect all trainable parameters
        params = []

        ########################################################################
        # Sub Modules
        ########################################################################

        ## Visual Encoder
        # Choose visual encoder architecture based on model_name
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        if model_name == "resnet18":
            self.outdim = 512  # Output dimension for ResNet18
            self.encoder = torchvision.models.resnet18(
                pretrained=self.pretrained
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
        elif model_name == "resnet34":
            self.outdim = 512  # Output dimension for ResNet34
            self.encoder = torchvision.models.resnet34(
                pretrained=self.pretrained
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
        elif model_name == "resnet50":
            self.outdim = 2048  # Output dimension for ResNet50
            self.encoder = torchvision.models.resnet50(
                pretrained=self.pretrained
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
        elif model_name == "vit":
            # Vision Transformer from HuggingFace
            import timm

            self.outdim = 768
            self.encoder = timm.create_model(
                "vit_base_patch16_224.mae", pretrained=False
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
        elif model_name == "vit_mae_imagenet_base":
            # Vision Transformer with MAE pretraining
            import timm

            self.outdim = 768
            self.encoder = timm.create_model(
                "vit_base_patch16_224.mae", pretrained=True
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
        elif model_name == "dinov2_base":
            self.outdim = 768
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            )
            self.normlayer = transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
            print(f"Loaded pretrained weights for dinov2_base")

        # Remove the final classification layer and keep only features
        self.encoder.fc = Identity()
        self.encoder.train()  # Set to training mode

        # Apply LayerNorm fine-tuning if enabled
        if self.layernorm_finetune and "vit" in model_name:
            self._freeze_except_layernorm(self.encoder)
            # Only add LayerNorm parameters to optimizer
            params += self._get_layernorm_params(self.encoder)
            print(
                "LayerNorm fine-tuning enabled - freezing all parameters except LayerNorm"
            )
        else:
            params += list(
                self.encoder.parameters()
            )  # Add encoder parameters to optimizer

        ## FiLM Conditioning Network
        # Network to generate FiLM parameters from language embeddings
        if self.use_film_cond:
            film_hidden_dim = 512
            self.film_network = nn.Sequential(
                nn.Linear(
                    768, film_hidden_dim
                ),  # 768 is typical BERT embedding size
                nn.ReLU(),
                nn.Linear(film_hidden_dim, film_hidden_dim),
                nn.ReLU(),
                nn.Linear(
                    film_hidden_dim, 2 * self.outdim
                ),  # 2 * outdim for scale and bias
            ).to(self.device)
            params += list(self.film_network.parameters())

        ## Behavioral Cloning Policy Network
        # Only create BC policy if behavioral cloning loss weight > 0
        if self.bc_weight > 0.0:
            feature_dim = 50  # Intermediate feature dimension
            bc_hidden_dim = 512  # Hidden dimension for BC policy

            # Set action dimension based on whether to include rotation and gripper
            action_dim = 32  # 2D Position only

            # Feature trunk: projects visual features to intermediate representation
            self.bc_trunk = nn.Sequential(
                nn.Linear(self.outdim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            ).to(self.device)

            # Policy head: predicts actions from features
            self.bc_policy = nn.Sequential(
                nn.Linear(feature_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, bc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(bc_hidden_dim, action_dim),
            ).to(self.device)

            # Add BC network parameters to optimizer
            params += list(self.bc_trunk.parameters())
            params += list(self.bc_policy.parameters())

        ## Optimizer
        # Create Adam optimizer for all trainable parameters
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    def _freeze_except_layernorm(self, module):
        """
        Freeze all parameters in a module except LayerNorm parameters.

        Args:
            module: PyTorch module to freeze parameters for
        """
        for param in module.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm parameters
        for name, child in module.named_modules():
            if isinstance(child, nn.LayerNorm):
                for param in child.parameters():
                    param.requires_grad = True
                print(f"Unfrozen LayerNorm parameters in: {name}")

    def _get_layernorm_params(self, module):
        """
        Get all LayerNorm parameters from a module.

        Args:
            module: PyTorch module to extract LayerNorm parameters from

        Returns:
            List of LayerNorm parameters
        """
        layernorm_params = []
        for child in module.modules():
            if isinstance(child, nn.LayerNorm):
                layernorm_params.extend(list(child.parameters()))
        return layernorm_params

    def get_trainable_params_count(self):
        """
        Get the count of trainable parameters.

        Returns:
            Tuple of (total parameters, trainable parameters)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return total_params, trainable_params

    def print_trainable_params_info(self):
        """
        Print information about trainable parameters.
        """
        total_params, trainable_params = self.get_trainable_params_count()
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(
            f"Percentage trainable: {100 * trainable_params / total_params:.2f}%"
        )

        if self.layernorm_finetune:
            print(
                "LayerNorm fine-tuning mode - only LayerNorm parameters are trainable"
            )

    def generate_film_params(self, lang_embedding):
        """
        Generate FiLM conditioning parameters from language embedding.

        FiLM (Feature-wise Linear Modulation) applies learned affine transformations
        to visual features based on language instructions:
            h_conditioned = scale * h + bias

        Args:
            lang_embedding: Language embedding tensor of shape [batch_size, 768]

        Returns:
            FiLM parameters tensor of shape [batch_size, 2 * outdim]
            First half contains scale parameters, second half contains bias parameters

        Raises:
            ValueError: If FiLM conditioning is not enabled
        """
        if not self.use_film_cond:
            raise ValueError(
                "FiLM conditioning is not enabled. Set use_film_cond=True in __init__"
            )

        return self.film_network(lang_embedding)

    def forward(
        self, obs, num_ims=1, obs_shape=[3, 224, 224], lang_embedding=None
    ):
        """
        Forward pass through the VisualEncoder model.

        This method processes visual observations and optionally applies FiLM conditioning
        based on language embeddings to produce visual representations.

        Args:
            obs: Input observations tensor of shape [batch_size, channels, height, width]
            num_ims: Number of images (legacy parameter, kept for compatibility)
            obs_shape: Expected observation shape (legacy parameter)
            lang_embedding: Optional language embeddings for FiLM conditioning [batch_size, 768]

        Returns:
            Visual features tensor of shape [batch_size, outdim]
        """
        # Create preprocessing pipeline with normalization
        preprocess = nn.Sequential(
            self.normlayer,
        )

        # Normalize input from [0, 255] to [0, 1] range (consistent with R3M)
        obs = obs.float() / 255.0
        obs_p = preprocess(obs)

        # Extract visual features using the encoder
        h = self.encoder(obs_p)

        # Apply FiLM conditioning if enabled and language embeddings provided
        if self.use_film_cond and lang_embedding is not None:
            # Generate FiLM parameters from language embedding
            cond_embed = self.generate_film_params(lang_embedding)

            # Ensure cond_embed has the right shape: [batch_size, 2 * outdim]
            # First half is scale, second half is bias
            if cond_embed.shape[-1] != 2 * self.outdim:
                raise ValueError(
                    f"cond_embed should have shape [..., {2 * self.outdim}], got {cond_embed.shape}"
                )

            # Split FiLM parameters into scale and bias
            scale = cond_embed[..., : self.outdim]
            bias = cond_embed[..., self.outdim :]

            # Apply FiLM conditioning: h = scale * h + bias
            h = scale * h + bias

        return h

    def sim(self, tensor1, tensor2):
        """
        Compute similarity between two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor

        Returns:
            Similarity score
        """
        if self.l2dist:
            # Use negative L2 distance
            return -torch.norm(tensor1 - tensor2, dim=-1)
        else:
            # Use cosine similarity
            return self.cs(tensor1, tensor2)
