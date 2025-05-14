import torch
from typing import Tuple, Dict, Any

def make_fc(n_in: int, n_hidden: int, n_out: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a fully connected topology mask.
    
    Args:
        n_in: Number of input features
        n_hidden: Number of hidden units
        n_out: Number of output features
        
    Returns:
        Tuple of (adjacency_mask, metadata)
    """
    n_total = n_in + n_hidden + n_out
    
    # Create all-ones adjacency matrix
    adj_mask = torch.ones((n_total, n_total), dtype=torch.bool)
    
    # Create metadata
    meta = {
        "type": "fully_connected",
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "n_total": n_total,
        "sparsity": 0.0,  # Fully connected = no sparsity
        "avg_degree": n_total - 1,  # Each node connects to all others
        "avg_clustering": 1.0,  # Complete graph has max clustering
        "avg_path_length": 1.0,  # All nodes are directly connected
    }
    
    return adj_mask, meta 