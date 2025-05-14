import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import numpy as np
import seaborn as sns

def plot_connectivity(
    mask: torch.Tensor,
    title: str,
    path: str,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300
) -> None:
    """
    Plot the connectivity matrix as a heatmap.
    
    Args:
        mask: Binary adjacency matrix
        title: Plot title
        path: Output file path
        figsize: Figure size (width, height)
        dpi: Dots per inch for output image
    """
    plt.figure(figsize=figsize)
    sns.heatmap(mask.cpu().numpy(), cmap='Blues')
    plt.title(title)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_network(
    mask: torch.Tensor,
    title: str,
    path: str,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300,
    node_size: int = 100,
    node_color: str = "lightblue",
    edge_color: str = "gray",
    alpha: float = 0.6
) -> None:
    """
    Plot the network using NetworkX spring layout.
    
    Args:
        mask: Binary adjacency matrix
        title: Plot title
        path: Output file path
        figsize: Figure size (width, height)
        dpi: Dots per inch for output image
        node_size: Size of nodes in the plot
        node_color: Color of nodes
        edge_color: Color of edges
        alpha: Transparency of edges
    """
    # Create directed graph
    G = nx.from_numpy_array(mask.cpu().numpy(), create_using=nx.DiGraph)
    
    # Use circular layout for better visualization of small-world structure
    pos = nx.circular_layout(G)
    
    plt.figure(figsize=figsize)
    
    # Draw edges with different colors for local vs long-range connections
    local_edges = []
    long_range_edges = []
    for u, v in G.edges():
        # Consider edges between nodes that are close in the circular layout as local
        if np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) < 0.5:
            local_edges.append((u, v))
        else:
            long_range_edges.append((u, v))
    
    # Draw local edges in blue
    nx.draw_networkx_edges(
        G, pos,
        edgelist=local_edges,
        edge_color='blue',
        alpha=alpha,
        arrows=True,
        arrowsize=10
    )
    
    # Draw long-range edges in red
    nx.draw_networkx_edges(
        G, pos,
        edgelist=long_range_edges,
        edge_color='red',
        alpha=alpha,
        arrows=True,
        arrowsize=10
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color
    )
    
    plt.title(title)
    plt.axis("off")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_degree_distribution(
    mask: torch.Tensor,
    title: str,
    path: str,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300
) -> None:
    """
    Plot the degree distribution of the network.
    
    Args:
        mask: Binary adjacency matrix
        title: Plot title
        path: Output file path
        figsize: Figure size (width, height)
        dpi: Dots per inch for output image
    """
    G = nx.from_numpy_array(mask.cpu().numpy(), create_using=nx.DiGraph)
    degrees = [d for n, d in G.degree()]
    
    plt.figure(figsize=figsize)
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), alpha=0.7)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Plot comparison of network metrics across different topologies.
    
    Args:
        metrics: Dictionary mapping topology names to their metrics
        output_path: Path to save the plot
    """
    # Extract metrics to plot
    topologies = list(metrics.keys())
    densities = [stats.get('density', 0.0) for stats in metrics.values()]
    clustering = [stats.get('avg_clustering', 0.0) for stats in metrics.values()]
    path_lengths = [stats.get('avg_path_length', 0.0) for stats in metrics.values()]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot density
    ax1.bar(topologies, densities)
    ax1.set_title('Network Density')
    ax1.set_ylim(0, 0.2)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot clustering
    ax2.bar(topologies, clustering)
    ax2.set_title('Average Clustering')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot path length
    ax3.bar(topologies, path_lengths)
    ax3.set_title('Average Path Length')
    ax3.set_ylim(0, 5.0)
    ax3.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 