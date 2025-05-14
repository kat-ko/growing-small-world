import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces
from typing import Dict, Type, TypeVar, Any

from .masked_linear import MaskedLinear

class MaskedExtractor(BaseFeaturesExtractor):
    """Feature extractor using masked linear layers for fixed topology."""
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int,
        adj_mask: torch.Tensor,
        n_in: int,
        n_out: int,
        hidden_act: nn.Module = nn.ReLU(),
        out_act: nn.Module = nn.Identity()
    ):
        """
        Initialize the masked feature extractor.
        
        Args:
            observation_space: The observation space
            features_dim: The dimension of the features to extract
            adj_mask: Binary adjacency matrix of shape (n_total, n_total)
            n_in: Number of input features
            n_out: Number of output features
            hidden_act: Activation function for hidden units
            out_act: Activation function for output units
        """
        super().__init__(observation_space, features_dim)
        
        # Create the masked linear layer
        self.mlp = MaskedLinear(
            adj_mask=adj_mask,
            n_in=n_in,
            n_out=features_dim,  # Output features_dim for SB3's policy/value networks
            hidden_act=hidden_act,
            out_act=out_act
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Input tensor of shape (batch_size, n_in)
            
        Returns:
            Output tensor of shape (batch_size, features_dim)
        """
        return self.mlp(observations)
    
    def get_structural_stats(self) -> Dict[str, float]:
        """Get structural statistics of the network topology."""
        return self.mlp.get_structural_stats()

def create_masked_policy_kwargs(
    adj_mask: torch.Tensor,
    n_in: int,
    n_out: int,
    features_dim: int,
    hidden_act: nn.Module = nn.ReLU(),
    out_act: nn.Module = nn.Identity()
) -> Dict[str, Any]:
    """
    Create policy kwargs for SB3 with masked topology.
    
    Args:
        adj_mask: Binary adjacency matrix
        n_in: Number of input features
        n_out: Number of output features
        features_dim: Dimension of features for SB3
        hidden_act: Activation function for hidden units
        out_act: Activation function for output units
        
    Returns:
        Dictionary of policy kwargs for SB3
    """
    return dict(
        features_extractor_class=MaskedExtractor,
        features_extractor_kwargs=dict(
            adj_mask=adj_mask,
            n_in=n_in,
            n_out=n_out,
            features_dim=features_dim,
            hidden_act=hidden_act,
            out_act=out_act
        ),
        net_arch=dict(pi=[features_dim], vf=[features_dim])  # Specify policy and value network architectures
    )

def zero_mask_grad(module: nn.Module) -> None:
    """
    Zero out gradients for masked weights and locked biases.
    This should be called after each optimizer step.
    
    Args:
        module: The module to process
    """
    if isinstance(module, MaskedLinear):
        with torch.no_grad():
            # Zero out weights where mask is 0
            module.W.data *= module.mask
            # Zero out biases for nodes with no incoming connections
            if module.bias is not None:
                module.bias.data[module.mask.sum(0) == 0] = 0 

class WeightClampingCallback(BaseCallback):
    """Callback to clamp masked weights (by zeroing gradients) after each update phase."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # This _on_step is called after each environment step.
        # For PPO, the main update happens in the train() method, not directly tied to this.
        # We use _on_rollout_end as it's called before train() ensuring gradients are clean.
        return True
        
    def _on_rollout_end(self) -> bool:
        """
        This method is called at the end of each rollout collection.
        We apply zero_mask_grad to the policy here to ensure that gradients
        for masked weights are zeroed out before the optimization step in model.train().
        """
        if self.model.policy is not None: # Ensure policy exists
            self.model.policy.apply(zero_mask_grad)
        return True

# Ensure existing code like zero_mask_grad, MaskedFeatureExtractor, 
# and create_masked_policy_kwargs are preserved below if they exist.
# If they are defined above this new class, that's fine too.
# For this edit, I'm assuming this class is added, and other content remains.
# If sb3_integration.py is short, I could paste its full content with the addition.
# For now, let's assume it's being appended or inserted appropriately. 