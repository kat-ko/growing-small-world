from typing import Tuple, Dict, Optional
import numpy as np
import torch

def make_random(
    n_in: int,
    n_hidden: int,
    n_out: int,
    density: float = 0.1,
    seed: Optional[int] = None,
    target_n_weights: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Generate a random network topology."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print("\nInitializing random network generation:")
    print(f"- Input nodes: {n_in}")
    print(f"- Hidden nodes: {n_hidden}")
    print(f"- Output nodes: {n_out}")
    print(f"- Target density: {density}")
    if target_n_weights is not None:
        print(f"- Target weights: {target_n_weights}")
    
    # Create adjacency matrix
    n_total = n_in + n_hidden + n_out
    adj = torch.zeros((n_total, n_total))
    
    # Generate random connections for hidden layer
    hidden_mask = torch.rand((n_hidden, n_hidden)) < density
    torch.diagonal(hidden_mask).fill_(0)  # Remove self-loops
    adj[n_in:n_in+n_hidden, n_in:n_in+n_hidden] = hidden_mask
    
    # Wire inputs and outputs
    adj = _wire_inputs_outputs(adj, n_in, n_hidden, n_out)
    
    # Ensure IO connectivity
    adj = ensure_io_stubs(adj, n_in, n_hidden, n_out)
    
    # Ensure minimum degree
    adj = ensure_min_degree(adj, min_in=2, min_out=2)
    
    # Pad to target weight budget if specified
    if target_n_weights is not None:
        adj = pad_to_budget(adj, target_n_weights)
    
    # Get network statistics
    stats = get_network_stats(adj, n_in, n_hidden, n_out)
    stats['type'] = 'random'
    
    # Add structural statistics
    stats.update(get_structural_stats(adj, n_in, n_hidden, n_out))
    
    print("\nNetwork generation complete!")
    print(f"Average clustering: {stats['avg_clustering']:.3f}")
    print(f"Average path length: {stats.get('avg_path_length', float('inf')):.3f}")
    print(f"Average degree: {stats['avg_degree']:.1f}")
    if target_n_weights is not None:
        print(f"Total weights: {stats['n_weights']}")
    
    return adj, stats 