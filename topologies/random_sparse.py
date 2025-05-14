import torch
import networkx as nx
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .utils import calculate_density, get_network_stats, validate_network, _wire_inputs_outputs, ensure_io_stubs, ensure_min_degree

def make_rs(
    n_in: int,
    n_hidden: int,
    n_out: int,
    density: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a random sparse topology.
    
    Args:
        n_in: Number of input features
        n_hidden: Number of hidden units
        n_out: Number of output features
        density: Target edge density
        seed: Random seed
        
    Returns:
        Tuple of (adjacency_mask, metadata)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    n_total = n_in + n_hidden + n_out
    adj_mask = torch.zeros(n_total, n_total, dtype=torch.bool)
    
    # Create random connections between input and hidden
    n_edges = int(density * n_in * n_hidden)
    src_indices = torch.randint(0, n_in, (n_edges,))
    dst_indices = torch.randint(n_in, n_in + n_hidden, (n_edges,))
    adj_mask[src_indices, dst_indices] = True
    
    # Create random connections between hidden and output
    n_edges = int(density * n_hidden * n_out)
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
        "density": density,
        "active_hidden": int(adj_mask[n_in:-n_out, :].any(dim=0).sum()),
        "effective_density": float(adj_mask[:n_in, n_in:-n_out].sum() / (n_in * n_hidden)),
        "output_density": float(adj_mask[n_in:-n_out, -n_out:].sum() / (n_hidden * n_out))
    }
    
    # Get network statistics
    stats = get_network_stats(adj_mask, n_in, n_hidden, n_out)
    meta.update(stats)
    
    # Validate network
    is_valid, error_msg = validate_network(adj_mask, n_total, density)
    if not is_valid:
        print(f"Warning: {error_msg}")
    
    meta["type"] = "random_sparse"
    meta["n_in"] = n_in
    meta["n_hidden"] = n_hidden
    meta["n_out"] = n_out
    meta["n_total"] = n_total
    meta["density"] = stats["density"]
    meta["avg_degree"] = stats["avg_degree"]
    meta["avg_clustering"] = stats["avg_clustering"]
    
    if "avg_path_length" in stats:
        meta["avg_path_length"] = stats["avg_path_length"]
    
    print("\nNetwork generation complete!")
    print(f"Network density: {stats['density']:.3f}")
    print(f"Average clustering: {stats['avg_clustering']:.3f}")
    print(f"Average path length: {stats.get('avg_path_length', 'inf')}")
    print(f"Average degree: {stats['avg_degree']:.1f}")
    
    return adj_mask, meta 