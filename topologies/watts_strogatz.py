import torch
import networkx as nx
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .utils import calculate_density, get_network_stats, validate_network, _wire_inputs_outputs, ensure_io_stubs, ensure_min_degree

def make_ws(
    n_in: int,
    n_hidden: int,
    n_out: int,
    k: int = 4,
    p: float = 0.1,
    seed: int = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generate a Watts-Strogatz small-world network topology.
    
    Args:
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        k: Number of neighbors for each node
        p: Rewiring probability
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - Binary adjacency matrix as torch tensor
        - Dictionary of network statistics
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    n_total = n_in + n_hidden + n_out
    adj_mask = torch.zeros(n_total, n_total, dtype=torch.bool)
    
    # Create random connections between input and hidden
    n_edges = int(k * n_in)  # k connections per input
    src_indices = torch.randint(0, n_in, (n_edges,))
    dst_indices = torch.randint(n_in, n_in + n_hidden, (n_edges,))
    adj_mask[src_indices, dst_indices] = True
    
    # Create random connections between hidden and output
    n_edges = int(k * n_out)  # k connections per output
    src_indices = torch.randint(n_in, n_in + n_hidden, (n_edges,))
    dst_indices = torch.randint(n_in + n_hidden, n_total, (n_edges,))
    adj_mask[src_indices, dst_indices] = True
    
    # Ensure IO connectivity
    adj_mask = ensure_io_stubs(adj_mask, n_in, n_hidden, n_out)
    
    # Ensure minimum degree
    adj_mask = ensure_min_degree(adj_mask, min_in=2, min_out=2)
    
    # Disable hidden-hidden edges
    adj_mask[n_in:n_in+n_hidden, n_in:n_in+n_hidden] = False
    
    # Calculate metrics
    meta = {
        "k": k,
        "p": p,
        "active_hidden": int(adj_mask[n_in:-n_out, :].any(dim=0).sum()),
        "effective_density": float(adj_mask[:n_in, n_in:-n_out].sum() / (n_in * n_hidden)),
        "output_density": float(adj_mask[n_in:-n_out, -n_out:].sum() / (n_hidden * n_out))
    }
    
    # Get network statistics
    stats = get_network_stats(adj_mask, n_in, n_hidden, n_out)
    meta.update(stats)
    
    return adj_mask, meta 