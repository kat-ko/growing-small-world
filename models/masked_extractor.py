import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MaskedExtractor(nn.Module):
    def __init__(self, adj_mask: torch.Tensor, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        # Create masked linear layers
        self.layer1 = MaskedLinear(n_in, n_hidden, adj_mask[:n_in, n_in:n_in+n_hidden])
        self.layer2 = MaskedLinear(n_hidden, n_out, adj_mask[n_in:n_in+n_hidden, n_in+n_hidden:])
        
        # Store weight statistics
        self.n_weights = int(adj_mask.sum())
        self.sparsity = 1.0 - (self.n_weights / (n_in + n_hidden + n_out) ** 2)
        
        # Calculate fan-in statistics
        fan_in_hidden = adj_mask[:n_in, n_in:n_in+n_hidden].sum(dim=0)
        fan_in_out = adj_mask[n_in:n_in+n_hidden, n_in+n_hidden:].sum(dim=0)
        self.fan_in_hidden_mean = float(fan_in_hidden.mean())
        self.fan_in_hidden_std = float(fan_in_hidden.std())
        self.fan_in_out_mean = float(fan_in_out.mean())
        self.fan_in_out_std = float(fan_in_out.std())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x), dim=-1)
        return x
    
    def get_stats(self) -> Dict[str, float]:
        """Get network statistics."""
        return {
            'n_weights': self.n_weights,
            'sparsity': self.sparsity,
            'fan_in_hidden_mean': self.fan_in_hidden_mean,
            'fan_in_hidden_std': self.fan_in_hidden_std,
            'fan_in_out_mean': self.fan_in_out_mean,
            'fan_in_out_std': self.fan_in_out_std
        } 