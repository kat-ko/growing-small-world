import pytest
import torch
import networkx as nx
from topologies.small_world_neat import make_sw_neat

def test_sw_neat_basic():
    """Test basic NEAT small-world network generation."""
    # Use small population and few generations for quick testing
    adj, meta = make_sw_neat(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        target_clustering=0.6,
        target_path_length=2.0,
        n_generations=2,
        pop_size=4
    )
    
    # Extract core graph (hidden nodes only)
    core = adj[4:-2, 4:-2]
    G_core = nx.from_numpy_array(core.numpy())
    
    # Verify metrics
    assert 0.09 <= meta["core_density"] <= 0.11, f"Core density {meta['core_density']:.3f} outside target range"
    assert meta["core_avg_clustering"] >= 0.40, f"Clustering {meta['core_avg_clustering']:.3f} below minimum"
    assert meta["core_avg_path_length"] <= 2.5, f"Path length {meta['core_avg_path_length']:.3f} above maximum"
    assert nx.is_connected(G_core), "Core graph is not connected"
    
    # Verify I/O wiring
    assert adj[:4, 4:-2].sum() == 8, "Incorrect number of input connections"  # 4 inputs × 2 connections
    assert adj[4:-2, -2:].sum() == 4, "Incorrect number of output connections"  # 2 outputs × 2 connections

def test_sw_neat_metrics():
    """Test NEAT small-world network metrics in detail."""
    adj, meta = make_sw_neat(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        target_clustering=0.6,
        target_path_length=2.0,
        n_generations=2,
        pop_size=4
    )
    
    # Verify all required metrics are present
    required_metrics = {
        'n_in', 'n_hidden', 'n_out', 'n_total',
        'density', 'avg_degree', 'avg_clustering', 'avg_path_length',
        'core_density', 'core_avg_degree', 'core_avg_clustering', 'core_avg_path_length'
    }
    assert all(metric in meta for metric in required_metrics), "Missing required metrics"
    
    # Verify metric types
    assert all(isinstance(meta[m], (int, float)) for m in required_metrics), "Invalid metric types"
    
    # Verify metric ranges
    assert 0 <= meta['density'] <= 1, "Invalid density"
    assert 0 <= meta['avg_clustering'] <= 1, "Invalid clustering"
    assert meta['avg_path_length'] > 0, "Invalid path length"
    assert meta['avg_degree'] >= 0, "Invalid average degree"

def test_sw_neat_consistency():
    """Test NEAT small-world network generation consistency."""
    # Generate two networks with same parameters
    adj1, meta1 = make_sw_neat(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        n_generations=2,
        pop_size=4,
        seed=42
    )
    
    adj2, meta2 = make_sw_neat(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        n_generations=2,
        pop_size=4,
        seed=42
    )
    
    # Verify same seed produces same results
    assert torch.all(adj1 == adj2), "Networks differ with same seed"
    assert meta1 == meta2, "Metrics differ with same seed"
    
    # Generate with different seed
    adj3, meta3 = make_sw_neat(
        n_in=4,
        n_hidden=64,
        n_out=2,
        density=0.10,
        n_generations=2,
        pop_size=4,
        seed=43
    )
    
    # Verify different seed produces different results
    assert not torch.all(adj1 == adj3), "Networks identical with different seeds"
    assert meta1 != meta3, "Metrics identical with different seeds" 