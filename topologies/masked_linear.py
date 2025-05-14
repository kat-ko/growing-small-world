import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

class MaskedLinear(nn.Module):
    """A linear layer with a fixed topology mask."""
    
    def __init__(
        self,
        adj_mask: torch.Tensor,
        n_in: int,
        n_out: int,
        hidden_act: nn.Module = nn.ReLU(),
        out_act: nn.Module = nn.Identity(),
        bias: bool = True
    ):
        """
        Initialize a masked linear layer.
        
        Args:
            adj_mask: Binary adjacency matrix of shape (n_total, n_total)
            n_in: Number of input features
            n_out: Number of output features
            hidden_act: Activation function for hidden units
            out_act: Activation function for output units
            bias: Whether to include bias terms
        """
        super().__init__()
        
        # Register the mask as a buffer (not a parameter)
        self.register_buffer("mask", adj_mask.float())
        
        # Initialize weights and bias
        self.W = nn.Parameter(torch.randn_like(self.mask))
        self.bias = nn.Parameter(torch.zeros(adj_mask.size(0))) if bias else None
        
        # Zero out masked weights
        with torch.no_grad():
            self.W.data *= self.mask
            if self.bias is not None:
                self.bias.data[self.mask.sum(0) == 0] = 0
        
        self.n_in = n_in
        self.n_out = n_out
        self.h_act = hidden_act
        self.out_act = out_act
        
        # Register backward hook to mask gradients
        self.register_full_backward_hook(self._mask_gradients)
        
    @property
    def weight(self):
        return self.W

    def _mask_gradients(self, module, grad_input, grad_output):
        """Mask gradients to respect the topology."""
        with torch.no_grad():
            # Mask weight gradients
            if self.W.grad is not None:
                self.W.grad.data *= self.mask
            # Mask bias gradients for nodes with no incoming connections
            if self.bias is not None and self.bias.grad is not None:
                self.bias.grad.data[self.mask.sum(0) == 0] = 0
        return grad_input
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the masked linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, n_in)
            
        Returns:
            Output tensor of shape (batch_size, n_out)
        """
        # Pad input with zeros for hidden and output units
        z = torch.zeros(x.size(0), self.mask.size(0), device=x.device)
        z[:, :self.n_in] = x
        
        # Apply masked linear transformation
        z = (z @ (self.W * self.mask).t())
        if self.bias is not None:
            z = z + self.bias
            
        # Apply activations
        z = self.h_act(z)
        return self.out_act(z[:, -self.n_out:])
    
    def get_structural_stats(self) -> Dict[str, float]:
        """Compute structural statistics of the network topology."""
        import networkx as nx
        
        # Convert mask to numpy and create directed graph
        G = nx.from_numpy_array(self.mask.cpu().numpy(), create_using=nx.DiGraph)
        G_undirected = G.to_undirected()
        
        # Compute statistics
        stats = {
            "sparsity": 1 - self.mask.float().mean().item(),
            "avg_clustering": nx.average_clustering(G_undirected),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
        
        # Add path length if graph is connected
        if nx.is_connected(G_undirected):
            stats["avg_path_length"] = nx.average_shortest_path_length(G_undirected)
            
        return stats 