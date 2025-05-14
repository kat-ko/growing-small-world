import torch
import networkx as nx
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .utils import calculate_density, get_network_stats, validate_network, _wire_inputs_outputs

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
    n_total = n_in + n_hidden + n_out
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print(f"\nInitializing random sparse network:")
    print(f"- Input nodes: {n_in}")
    print(f"- Hidden nodes: {n_hidden}")
    print(f"- Output nodes: {n_out}")
    print(f"- Target density: {density}")
    
    # Calculate target number of edges
    target_edges = int(density * n_total * (n_total - 1))
    
    # Create initial random graph
    print("\nGenerating random graph...")
    adj_matrix = np.zeros((n_total, n_total), dtype=bool)
    
    # Add random edges for remaining connections
    possible_edges = [(i, j) for i in range(n_total) for j in range(n_total) 
                     if i != j and not adj_matrix[i, j]]
    np.random.shuffle(possible_edges)
    
    # Add remaining edges
    for i, j in possible_edges[:target_edges]:
        adj_matrix[i, j] = True
    
    print(f"Adding {target_edges} random edges...")
    
    # Convert to torch tensor
    adj_mask = torch.from_numpy(adj_matrix)
    
    # Wire inputs and outputs with controlled fan-in/out
    adj_mask = _wire_inputs_outputs(adj_mask, n_in, n_hidden, n_out)
    
    # Get network statistics on core graph (hidden nodes only)
    core = adj_mask[n_in:-n_out, n_in:-n_out]
    stats = get_network_stats(core)
    
    # Validate network
    is_valid, error_msg = validate_network(adj_mask, n_total, density)
    if not is_valid:
        print(f"Warning: {error_msg}")
    
    # Create metadata
    meta = {
        "type": "random_sparse",
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "n_total": n_total,
        "density": stats["density"],
        "avg_degree": stats["avg_degree"],
        "avg_clustering": stats["avg_clustering"]
    }
    
    if "avg_path_length" in stats:
        meta["avg_path_length"] = stats["avg_path_length"]
    
    print("\nNetwork generation complete!")
    print(f"Network density: {stats['density']:.3f}")
    print(f"Average clustering: {stats['avg_clustering']:.3f}")
    print(f"Average path length: {stats.get('avg_path_length', 'inf')}")
    print(f"Average degree: {stats['avg_degree']:.1f}")
    
    return adj_mask, meta 