import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import networkx as nx
import yaml

from topologies.fully_connected import make_fc
from topologies.random_sparse import make_rs
from topologies.small_world_neat import make_sw_neat
from topologies.watts_strogatz import make_ws
from topologies.modular import make_modular
from topologies.viz import plot_connectivity, plot_network, plot_degree_distribution, plot_metrics_comparison
from topologies.utils import get_network_stats

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def get_core_stats(adj_mask, n_in, n_out):
    """Get statistics for the core (hidden) part of the network."""
    core = adj_mask[n_in:-n_out, n_in:-n_out]
    n_hidden = core.size(0)
    return get_network_stats(core, n_hidden)

def run_comparison(cfg: DictConfig) -> None:
    """Run topology comparison and save results."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, f"topology_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Generate topologies
    topologies = {
        "Fully Connected": make_fc(
            n_in=cfg.n_in,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out
        ),
        "Random Sparse": make_rs(
            n_in=cfg.n_in,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out,
            density=cfg.target_density
        ),
        "Small World NEAT": make_sw_neat(
            n_in=cfg.n_in,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out,
            density=cfg.target_density,
            target_clustering=cfg.target_clustering,
            target_path_length=cfg.target_path_length,
            pop_size=cfg.pop_size,
            n_generations=cfg.n_generations
        ),
        "Small World WS": make_ws(
            n_in=cfg.n_in,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out
        ),
        "Modular": make_modular(
            n_in=cfg.n_in,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out,
            density=cfg.target_density
        )
    }
    
    # Process each topology
    metrics = {}
    for name, (adj_matrix, metadata) in topologies.items():
        print(f"\nProcessing {name} topology...")
        
        # Create topology directory
        topology_dir = os.path.join(output_dir, name.lower().replace(" ", "_"))
        os.makedirs(topology_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(topology_dir, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)
        
        # Generate visualizations
        plot_connectivity(
            adj_matrix,
            f"{name} Connectivity",
            os.path.join(topology_dir, "connectivity.png")
        )
        
        plot_network(
            adj_matrix,
            f"{name} Network",
            os.path.join(topology_dir, "network.png")
        )
        
        plot_degree_distribution(
            adj_matrix,
            f"{name} Degree Distribution",
            os.path.join(topology_dir, "degree_dist.png")
        )
        
        # Store metrics
        metrics[name] = metadata
    
    # Save metrics comparison
    plot_metrics_comparison(metrics, os.path.join(output_dir, "metrics_comparison.png"))
    
    print(f"\nComparison complete! Results saved to: {output_dir}")
    
    # Print summary
    print("\nTopology Metrics Summary:")
    print("-" * 50)
    for name, stats in metrics.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_comparison(cfg)

if __name__ == "__main__":
    main() 