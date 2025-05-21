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

def calculate_clustering(adj_matrix: torch.Tensor) -> float:
    """Calculate average clustering coefficient for the network."""
    G = nx.from_numpy_array(adj_matrix.cpu().numpy(), create_using=nx.DiGraph)
    G_undirected = G.to_undirected()
    return float(nx.average_clustering(G_undirected))

def calculate_path_length(adj_matrix: torch.Tensor) -> float:
    """Calculate average shortest path length for the network."""
    G = nx.from_numpy_array(adj_matrix.cpu().numpy(), create_using=nx.DiGraph)
    G_undirected = G.to_undirected()
    try:
        return float(nx.average_shortest_path_length(G_undirected))
    except nx.NetworkXError:
        return float('inf')  # Return infinity if graph is not connected

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

def ensure_io_stubs(adj: torch.Tensor, n_in: int, n_hidden: int, n_out: int) -> torch.Tensor:
    """Ensure every hidden unit has at least one input and one output connection."""
    # INPUT → hidden
    for h in range(n_hidden):
        col = n_in + h
        if adj[:n_in, col].sum() == 0:
            src = np.random.randint(0, n_in)
            adj[src, col] = True
    
    # hidden → OUTPUT
    for h in range(n_hidden):
        row = n_in + h
        if adj[row, -n_out:].sum() == 0:
            dst = -np.random.randint(1, n_out + 1)  # pick a random output
            adj[row, dst] = True
    return adj

# Deprecated: Use ensure_io_stubs instead
def ensure_input_stubs(adj: torch.Tensor, n_in: int, n_hidden: int) -> torch.Tensor:
    """Deprecated: Use ensure_io_stubs instead."""
    print("Warning: ensure_input_stubs is deprecated. Use ensure_io_stubs instead.")
    return ensure_io_stubs(adj, n_in, n_hidden, 0)  # Pass 0 for n_out to only do input stubs 

def ensure_min_degree(adj: torch.Tensor, min_in: int = 2, min_out: int = 2) -> torch.Tensor:
    """Ensure each node has at least min_in incoming and min_out outgoing edges."""
    n = adj.shape[0]
    
    # Check and fix incoming edges
    for j in range(n):
        in_degree = int(adj[:, j].sum().item())  # Convert to Python integer
        if in_degree < min_in:
            # Add random incoming edges
            n_missing = int(min_in - in_degree)  # Ensure integer
            candidates = [i for i in range(n) if i != j and not adj[i, j].item()]
            if candidates:
                for i in np.random.choice(candidates, size=min(n_missing, len(candidates)), replace=False):
                    adj[i, j] = True
    
    # Check and fix outgoing edges
    for i in range(n):
        out_degree = int(adj[i, :].sum().item())  # Convert to Python integer
        if out_degree < min_out:
            # Add random outgoing edges
            n_missing = int(min_out - out_degree)  # Ensure integer
            candidates = [j for j in range(n) if j != i and not adj[i, j].item()]
            if candidates:
                for j in np.random.choice(candidates, size=min(n_missing, len(candidates)), replace=False):
                    adj[i, j] = True
    
    return adj 

def pad_to_budget(adj_mask: torch.Tensor, target_n_weights: int) -> torch.Tensor:
    """
    Pad network to target weight budget while preserving topology characteristics.
    
    Args:
        adj_mask: Binary adjacency matrix
        target_n_weights: Target number of weights
        
    Returns:
        Padded adjacency matrix
    """
    current_weights = int(adj_mask.sum())
    if current_weights >= target_n_weights:
        return adj_mask
    
    # Calculate number of weights to add
    n_weights_to_add = target_n_weights - current_weights
    
    # Get current network properties
    n_total = adj_mask.shape[0]
    n_in = int(n_total * 0.1)  # Assuming 10% input nodes
    n_out = int(n_total * 0.1)  # Assuming 10% output nodes
    n_hidden = n_total - n_in - n_out
    
    # Calculate current densities
    input_density = float(adj_mask[:n_in, n_in:-n_out].sum() / (n_in * n_hidden))
    output_density = float(adj_mask[n_in:-n_out, -n_out:].sum() / (n_hidden * n_out))
    
    # Calculate target densities based on current ratios
    total_possible_weights = n_in * n_hidden + n_hidden * n_out
    target_input_weights = int(n_weights_to_add * (n_in * n_hidden) / total_possible_weights)
    target_output_weights = n_weights_to_add - target_input_weights
    
    # Add weights to input connections
    if target_input_weights > 0:
        input_mask = ~adj_mask[:n_in, n_in:-n_out]
        input_indices = torch.nonzero(input_mask)
        if len(input_indices) > 0:
            n_input_edges = min(target_input_weights, len(input_indices))
            selected_indices = torch.randperm(len(input_indices))[:n_input_edges]
            for idx in selected_indices:
                i, j = input_indices[idx]
                adj_mask[i, j + n_in] = True
    
    # Add weights to output connections
    if target_output_weights > 0:
        output_mask = ~adj_mask[n_in:-n_out, -n_out:]
        output_indices = torch.nonzero(output_mask)
        if len(output_indices) > 0:
            n_output_edges = min(target_output_weights, len(output_indices))
            selected_indices = torch.randperm(len(output_indices))[:n_output_edges]
            for idx in selected_indices:
                i, j = output_indices[idx]
                adj_mask[i + n_in, j + n_in + n_hidden] = True
    
    # Verify final weight count
    final_weights = int(adj_mask.sum())
    if final_weights != target_n_weights:
        print(f"Warning: Could not achieve exact target weight count. Got {final_weights} instead of {target_n_weights}")
    
    return adj_mask

def get_structural_stats(adj: torch.Tensor, n_in: int, n_hidden: int, n_out: int) -> Dict[str, float]:
    """Get detailed structural statistics about the network."""
    stats = {}
    
    # Weight counts
    stats['n_weights'] = int(adj.sum())
    
    # Fan-in statistics
    fan_in_hidden = adj[:n_in, n_in:n_in+n_hidden].sum(dim=0)
    fan_in_out = adj[n_in:n_in+n_hidden, n_in+n_hidden:].sum(dim=0)
    
    stats['fan_in_hidden_mean'] = float(fan_in_hidden.mean())
    stats['fan_in_hidden_std'] = float(fan_in_hidden.std())
    stats['fan_in_out_mean'] = float(fan_in_out.mean())
    stats['fan_in_out_std'] = float(fan_in_out.std())
    
    return stats 