import torch
import networkx as nx
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .utils import calculate_density, get_network_stats, validate_network, _wire_inputs_outputs

def make_ws(
    n_in: int,
    n_hidden: int,
    n_out: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generate a Watts-Strogatz small-world network topology.
    
    Args:
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - Binary adjacency matrix as torch tensor
        - Dictionary of network statistics
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Initialize parameters
    k = 6  # Changed from 8 to 6 to achieve density ~0.095
    beta = 0.1  # Rewiring probability
    
    # Create WS graph for hidden layer
    G = nx.watts_strogatz_graph(n_hidden, k, beta)
    
    # Convert to adjacency matrix
    hidden_adj = nx.to_numpy_array(G)
    
    # Create full adjacency matrix
    n_total = n_in + n_hidden + n_out
    adj = np.zeros((n_total, n_total))
    
    # Place hidden layer connectivity in the full matrix
    start_idx = n_in
    end_idx = start_idx + n_hidden
    adj[start_idx:end_idx, start_idx:end_idx] = hidden_adj
    
    # Convert to tensor
    adj_tensor = torch.from_numpy(adj).float()
    
    # Wire inputs and outputs with controlled fan-in/out
    adj_tensor = _wire_inputs_outputs(adj_tensor, n_in, n_hidden, n_out)
    
    # Get network statistics
    stats = get_network_stats(adj_tensor, n_in, n_hidden, n_out)
    
    # Add WS-specific parameters to metadata
    stats.update({
        'type': 'watts_strogatz',
        'k': float(k),
        'beta': float(beta)
    })
    
    return adj_tensor, stats 