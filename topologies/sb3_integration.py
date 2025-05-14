import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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