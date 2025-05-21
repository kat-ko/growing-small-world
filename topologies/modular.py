import torch
import networkx as nx
import numpy as np
from typing import Tuple, Dict, Any, Optional
import community.community_louvain as community
from .utils import (
    calculate_density, get_network_stats, validate_network, 
    _wire_inputs_outputs, ensure_io_stubs, ensure_min_degree,
    pad_to_budget, get_structural_stats
)

def make_modular(
    n_in: int,
    n_hidden: int,
    n_out: int,
    density: float = 0.1,
    seed: Optional[int] = None,
    p_intra: float = 0.8,
    p_inter: float = 0.008,
    n_modules: int = 6,
    max_retries: int = 20,
    target_n_weights: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generate a modular network topology.
    
    Args:
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        density: Target network density
        seed: Random seed for reproducibility
        p_intra: Probability of connections within modules
        p_inter: Initial probability of connections between modules
        n_modules: Number of modules to create
        max_retries: Maximum attempts to achieve target density and modularity
        target_n_weights: Target number of weights for the network
        
    Returns:
        Tuple of:
        - Binary adjacency matrix as torch tensor
        - Dictionary of network statistics
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print("\nInitializing modular network generation:")
    print(f"- Input nodes: {n_in}")
    print(f"- Hidden nodes: {n_hidden}")
    print(f"- Output nodes: {n_out}")
    print(f"- Target density: {density}")
    if target_n_weights is not None:
        print(f"- Target weights: {target_n_weights}")
    
    # Calculate module sizes
    module_size = n_hidden // n_modules
    remainder = n_hidden % n_modules
    module_sizes = [module_size + 1 if i < remainder else module_size for i in range(n_modules)]
    
    # Fine-tuning loop
    best_adj = None
    best_stats = None
    best_modularity = 0.0
    best_density_diff = float('inf')
    current_p_inter = p_inter
    last_valid_adj = None
    last_valid_stats = None
    
    for attempt in range(max_retries):
        # Create block matrix for hidden nodes
        hidden_adj = np.zeros((n_hidden, n_hidden))
        start_idx = 0
        
        # Fill diagonal blocks (intra-module connections)
        for size in module_sizes:
            end_idx = start_idx + size
            block = np.random.rand(size, size) < p_intra
            np.fill_diagonal(block, 0)  # Remove self-loops
            hidden_adj[start_idx:end_idx, start_idx:end_idx] = block
            start_idx = end_idx
        
        # Add inter-module connections
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                i_start = sum(module_sizes[:i])
                i_end = i_start + module_sizes[i]
                j_start = sum(module_sizes[:j])
                j_end = j_start + module_sizes[j]
                
                block = np.random.rand(module_sizes[i], module_sizes[j]) < current_p_inter
                hidden_adj[i_start:i_end, j_start:j_end] = block
                hidden_adj[j_start:j_end, i_start:i_end] = block.T
        
        # Create full adjacency matrix
        n_total = n_in + n_hidden + n_out
        adj = np.zeros((n_total, n_total))
        adj[n_in:n_in+n_hidden, n_in:n_in+n_hidden] = hidden_adj
        
        # Convert to tensor and wire inputs/outputs
        adj_tensor = torch.from_numpy(adj).float()
        adj_tensor = _wire_inputs_outputs(adj_tensor, n_in, n_hidden, n_out)
        
        # Ensure IO connectivity
        adj_tensor = ensure_io_stubs(adj_tensor, n_in, n_hidden, n_out)
        
        # Ensure minimum degree
        adj_tensor = ensure_min_degree(adj_tensor, min_in=2, min_out=2)
        
        # Get network statistics
        stats = get_network_stats(adj_tensor, n_in, n_hidden, n_out)
        
        # Calculate modularity
        G = nx.from_numpy_array(hidden_adj)
        try:
            partition = community.best_partition(G)
            modularity = community.modularity(partition, G)
        except:
            modularity = 0.0
        
        # Get core graph statistics
        core = adj_tensor[n_in:-n_out, n_in:-n_out]
        core_stats = get_network_stats(core)
        
        # Calculate density difference
        density_diff = abs(core_stats['density'] - density)
        
        # Store last valid network
        if modularity >= 0.60:
            last_valid_adj = adj_tensor.clone()
            last_valid_stats = stats.copy()
        
        # Check if this is the best solution so far
        is_better = False
        
        # If we have a solution meeting all criteria
        if (density_diff < 0.015 and modularity >= 0.60):
            if modularity > best_modularity:
                is_better = True
        # If we don't have a solution meeting all criteria yet
        elif best_adj is None:
            if modularity >= 0.60 and density_diff < best_density_diff:
                is_better = True
        
        if is_better:
            best_adj = adj_tensor
            best_stats = stats
            best_modularity = modularity
            best_density_diff = density_diff
            
            # Add modular-specific parameters to metadata
            best_stats.update({
                'type': 'modular',
                'modularity': float(modularity),
                'n_communities': float(len(set(partition.values()))),
                'p_intra': float(p_intra),
                'p_inter': float(current_p_inter)
            })
            
            print(f"\nFound better solution (attempt {attempt + 1}):")
            print(f"Core density: {core_stats['density']:.3f}")
            print(f"Modularity: {modularity:.3f}")
        
        # Adjust p_inter based on current density
        if core_stats['density'] > density:
            current_p_inter *= 0.9  # Too dense, reduce inter-module connections
        else:
            current_p_inter *= 1.1  # Too sparse, increase inter-module connections
    
    # Use best solution if found, otherwise use last valid network
    if best_adj is None:
        print("\nWarning: Could not find solution meeting all criteria.")
        if last_valid_adj is not None:
            print("Using last valid network with modularity >= 0.60.")
            best_adj = last_valid_adj
            best_stats = last_valid_stats
        else:
            print("No valid networks found. Using last generated network.")
            best_adj = adj_tensor
            best_stats = stats
        
        best_stats.update({
            'type': 'modular',
            'modularity': float(modularity),
            'n_communities': float(len(set(partition.values()))),
            'p_intra': float(p_intra),
            'p_inter': float(current_p_inter)
        })
        
        # Ensure minimum degrees even in fallback case
        best_adj = ensure_min_degree(best_adj, min_in=2, min_out=2)
    
    # Disable hidden-hidden edges (moved here to ensure it's applied to all paths)
    best_adj[n_in:n_in+n_hidden, n_in:n_in+n_hidden] = False
    
    # Verify hidden-hidden edges are disabled before proceeding
    assert best_adj[n_in:n_in+n_hidden, n_in:n_in+n_hidden].sum() == 0, "Hidden-hidden edges not disabled"
    
    # Pad to target weight budget if specified
    if target_n_weights is not None:
        best_adj = pad_to_budget(best_adj, target_n_weights)
        best_stats['n_weights'] = int(best_adj.sum())
    
    # Add structural statistics
    best_stats.update(get_structural_stats(best_adj, n_in, n_hidden, n_out))
    
    # Verify final network properties
    assert best_adj.sum() > 0, "Network has no edges"
    
    print("\nNetwork generation complete!")
    print(f"Number of communities: {len(set(partition.values()))}")
    print(f"Modularity score: {best_modularity:.3f}")
    print(f"Average clustering: {core_stats['avg_clustering']:.3f}")
    print(f"Average path length: {core_stats.get('avg_path_length', float('inf')):.3f}")
    print(f"Average degree: {core_stats['avg_degree']:.1f}")
    if target_n_weights is not None:
        print(f"Total weights: {best_stats['n_weights']}")
    
    return best_adj, best_stats 