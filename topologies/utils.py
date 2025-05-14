import torch
import networkx as nx
from typing import Tuple, Dict, Any
import numpy as np

def calculate_density(adj_matrix: torch.Tensor) -> float:
    """
    Calculate network density.
    
    Args:
        adj_matrix: Binary adjacency matrix
        
    Returns:
        Network density
    """
    n = len(adj_matrix)
    edges = adj_matrix.sum().item()
    possible_edges = n * (n - 1)  # Directed graph, no self-loops
    return edges / possible_edges if possible_edges > 0 else 0.0

def get_network_stats(adj_matrix: torch.Tensor, n_in: int = 0, n_hidden: int = None, n_out: int = 0) -> Dict[str, float]:
    """
    Calculate network statistics for both core and full graph.
    
    Args:
        adj_matrix: Binary adjacency matrix
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes (if None, assumes entire matrix is hidden)
        n_out: Number of output nodes
        
    Returns:
        Dictionary of network statistics
    """
    if n_hidden is None:
        n_hidden = len(adj_matrix)
        
    # Get core graph (hidden nodes only)
    if n_in > 0 or n_out > 0:
        core = adj_matrix[n_in:n_in+n_hidden, n_in:n_in+n_hidden]
    else:
        core = adj_matrix
        
    # Convert to NetworkX graphs
    G = nx.from_numpy_array(adj_matrix.cpu().numpy(), create_using=nx.DiGraph)
    G_undirected = G.to_undirected()
    
    G_core = nx.from_numpy_array(core.cpu().numpy(), create_using=nx.DiGraph)
    G_core_undirected = G_core.to_undirected()
    
    # Calculate statistics
    stats = {
        'n_in': float(n_in),
        'n_hidden': float(n_hidden),
        'n_out': float(n_out),
        'n_total': float(len(adj_matrix)),
        'density': float(calculate_density(adj_matrix)),
        'avg_degree': float(sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)),
        'avg_clustering': float(nx.average_clustering(G_undirected) if G.number_of_nodes() > 0 else 0.0),
    }
    
    try:
        stats['avg_path_length'] = float(nx.average_shortest_path_length(G_undirected))
    except:
        stats['avg_path_length'] = float('inf')
        
    # Add core statistics
    stats.update({
        'core_density': float(calculate_density(core)),
        'core_avg_degree': float(sum(dict(G_core.degree()).values()) / max(G_core.number_of_nodes(), 1)),
        'core_avg_clustering': float(nx.average_clustering(G_core_undirected) if G_core.number_of_nodes() > 0 else 0.0)
    })
    
    try:
        stats['core_avg_path_length'] = float(nx.average_shortest_path_length(G_core_undirected))
    except:
        stats['core_avg_path_length'] = float('inf')
        
    return stats

def validate_network(mask: torch.Tensor, n_total: int, target_density: float) -> Tuple[bool, str]:
    """
    Validate network properties.
    
    Args:
        mask: Binary adjacency matrix
        n_total: Total number of nodes
        target_density: Target network density
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    G = nx.from_numpy_array(mask.cpu().numpy(), create_using=nx.DiGraph)
    density = calculate_density(mask)
    
    if not nx.is_weakly_connected(G):
        return False, "Network is not weakly connected"
        
    if abs(density - target_density) >= 0.02:
        return False, f"Density {density:.3f} differs from target {target_density:.3f} by more than 2%"
        
    return True, ""

def _wire_inputs_outputs(adj_mask, n_in, n_hidden, n_out, fan_k=2):
    """
    Wire input and output nodes with controlled fan-in/out.
    
    Args:
        adj_mask: Binary adjacency matrix
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        fan_k: Number of connections per input/output node
        
    Returns:
        Modified adjacency matrix
    """
    # Wire inputs to hidden
    for i in range(n_in):
        # Connect to k random hidden nodes
        targets = np.random.choice(n_hidden, size=fan_k, replace=False)
        for j in targets:
            adj_mask[i, n_in + j] = True
    
    # Wire hidden to outputs
    for i in range(n_out):
        # Connect from k random hidden nodes
        sources = np.random.choice(n_hidden, size=fan_k, replace=False)
        for j in sources:
            adj_mask[n_in + j, n_in + n_hidden + i] = True
    
    return adj_mask 

def io_reachability(adj_mask: torch.Tensor, n_in: int, n_hidden: int, n_out: int) -> float:
    """Calculate the percentage of output nodes reachable from all input nodes."""
    # Ensure adj_mask is on CPU and converted to NumPy for NetworkX
    G = nx.from_numpy_array(adj_mask.cpu().numpy(), create_using=nx.DiGraph)
    
    inputs  = range(n_in)
    output_start_idx = n_in + n_hidden
    output_end_idx = n_in + n_hidden + n_out
    outputs = range(output_start_idx, output_end_idx)
    
    reachable_outputs_count = 0
    if n_out == 0: 
        return 0.0 
        
    for o_node in outputs:
        if o_node not in G:
            continue

        all_inputs_can_reach_output = True
        for i_node in inputs:
            if i_node not in G:
                all_inputs_can_reach_output = False
                break 
            
            if not nx.has_path(G, i_node, o_node):
                all_inputs_can_reach_output = False
                break 
        
        if all_inputs_can_reach_output:
            reachable_outputs_count += 1
            
    return reachable_outputs_count / n_out 