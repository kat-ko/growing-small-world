import torch
import networkx as nx
from typing import Tuple, Dict, Any, Optional
import numpy as np
from .utils import (
    calculate_density, get_network_stats, validate_network, 
    _wire_inputs_outputs, ensure_io_stubs, ensure_min_degree,
    pad_to_budget
)

def make_rs(
    n_in: int,
    n_hidden: int,
    n_out: int,
    density: float = 0.1,
    seed: Optional[int] = None,
    target_n_weights: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create a random sparse topology.
    
    Args:
        n_in: Number of input features
        n_hidden: Number of hidden units
        n_out: Number of output features
        density: Target edge density
        seed: Random seed
        target_n_weights: Target number of weights for the network
        
    Returns:
        Tuple of (adjacency_mask, metadata)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print("\nInitializing random sparse network generation:")
    print(f"- Input nodes: {n_in}")
    print(f"- Hidden nodes: {n_hidden}")
    print(f"- Output nodes: {n_out}")
    print(f"- Target density: {density}")
    if target_n_weights is not None:
        print(f"- Target weights: {target_n_weights}")
    
    n_total = n_in + n_hidden + n_out
    adj_mask = torch.zeros(n_total, n_total, dtype=torch.bool)
    
    # Calculate target edges for each connection type
    input_hidden_edges = int(density * n_in * n_hidden)
    hidden_output_edges = int(density * n_hidden * n_out)
    
    # Create random connections between input and hidden
    src_indices = torch.randint(0, n_in, (input_hidden_edges,))
    dst_indices = torch.randint(n_in, n_in + n_hidden, (input_hidden_edges,))
    adj_mask[src_indices, dst_indices] = True
    
    # Create random connections between hidden and output
    src_indices = torch.randint(n_in, n_in + n_hidden, (hidden_output_edges,))
    dst_indices = torch.randint(n_in + n_hidden, n_total, (hidden_output_edges,))
    adj_mask[src_indices, dst_indices] = True
    
    # Ensure IO connectivity
    adj_mask = ensure_io_stubs(adj_mask, n_in, n_hidden, n_out)
    
    # Ensure minimum degree
    adj_mask = ensure_min_degree(adj_mask, min_in=2, min_out=2)
    
    # Disable hidden-hidden edges
    adj_mask[n_in:n_in+n_hidden, n_in:n_in+n_hidden] = False
    
    # Calculate initial statistics
    stats = get_network_stats(adj_mask, n_in, n_hidden, n_out)
    current_density = stats['density']
    
    # If we need to pad to target weight budget
    if target_n_weights is not None:
        # Calculate how many weights we need to add
        current_weights = int(adj_mask.sum())
        weights_to_add = target_n_weights - current_weights
        
        if weights_to_add > 0:
            # Calculate target densities for input and output connections
            input_density = float(adj_mask[:n_in, n_in:-n_out].sum() / (n_in * n_hidden))
            output_density = float(adj_mask[n_in:-n_out, -n_out:].sum() / (n_hidden * n_out))
            
            # Add weights proportionally to maintain density ratios
            total_possible = n_in * n_hidden + n_hidden * n_out
            input_weights = int(weights_to_add * (n_in * n_hidden) / total_possible)
            output_weights = weights_to_add - input_weights
            
            # Add input connections
            if input_weights > 0:
                input_mask = ~adj_mask[:n_in, n_in:-n_out]
                input_indices = torch.nonzero(input_mask)
                if len(input_indices) > 0:
                    n_input_edges = min(input_weights, len(input_indices))
                    selected_indices = torch.randperm(len(input_indices))[:n_input_edges]
                    for idx in selected_indices:
                        i, j = input_indices[idx]
                        adj_mask[i, j + n_in] = True
            
            # Add output connections
            if output_weights > 0:
                output_mask = ~adj_mask[n_in:-n_out, -n_out:]
                output_indices = torch.nonzero(output_mask)
                if len(output_indices) > 0:
                    n_output_edges = min(output_weights, len(output_indices))
                    selected_indices = torch.randperm(len(output_indices))[:n_output_edges]
                    for idx in selected_indices:
                        i, j = output_indices[idx]
                        adj_mask[i + n_in, j + n_in + n_hidden] = True
    
    # Calculate final statistics
    stats = get_network_stats(adj_mask, n_in, n_hidden, n_out)
    
    # Create metadata
    meta = {
        "type": "random_sparse",
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "n_total": n_total,
        "density": stats["density"],
        "avg_degree": stats["avg_degree"],
        "avg_clustering": stats["avg_clustering"],
        "initial_density": current_density
    }
    
    if "avg_path_length" in stats:
        meta["avg_path_length"] = stats["avg_path_length"]
    
    if target_n_weights is not None:
        meta["n_weights"] = int(adj_mask.sum())
    
    print("\nNetwork generation complete!")
    print(f"Initial density: {current_density:.3f}")
    print(f"Final density: {stats['density']:.3f}")
    print(f"Average clustering: {stats['avg_clustering']:.3f}")
    print(f"Average path length: {stats.get('avg_path_length', 'inf')}")
    print(f"Average degree: {stats['avg_degree']:.1f}")
    if target_n_weights is not None:
        print(f"Total weights: {meta['n_weights']}")
    
    return adj_mask, meta 