import torch
import neat
import networkx as nx
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
import json
from .utils import calculate_density, get_network_stats, validate_network, _wire_inputs_outputs
import random

class SmallWorldFitness:
    """Fitness function for evolving small-world networks."""
    
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        target_density: float = 0.1,
        target_clustering: float = 0.6,
        target_path_length: float = 2.0
    ):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_total = n_in + n_hidden + n_out
        self.target_density = target_density
        self.target_clustering = target_clustering
        self.target_path_length = target_path_length
        self.target_edges = int(target_density * self.n_total * (self.n_total - 1))
        
    def __call__(self, genomes, config):
        """
        Evaluate fitness of genomes.
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
            
        Returns:
            None (fitness is set on genome objects)
        """
        for genome_id, genome in genomes:
            try:
                # Initialize empty genome with some connections
                if not genome.connections:
                    # Build a WS ring over hidden nodes with higher k to ensure density
                    k = 8  # k = 8 ⇒ density ≈ 0.09 for 64 nodes with 35% reverse edges
                    beta = 0.1  # lower rewiring probability for higher clustering
                    ws = nx.watts_strogatz_graph(self.n_hidden, k, beta, seed=None)
                    
                    # Add all edges from WS graph
                    for (u, v) in ws.edges():
                        src = int(self.n_in + u)  # shift into hidden index range
                        dst = int(self.n_in + v)
                        key = (src, dst)
                        conn = neat.genome.DefaultConnectionGene(key)
                        conn.weight = np.random.normal(
                            config.genome_config.weight_init_mean,
                            config.genome_config.weight_init_stdev
                        )
                        conn.enabled = True
                        genome.connections[key] = conn
                        
                        # Add reverse connection with 35% probability for better connectivity
                        if np.random.random() < 0.35:
                            key_rev = (dst, src)
                            conn_rev = neat.genome.DefaultConnectionGene(key_rev)
                            conn_rev.weight = np.random.normal(
                                config.genome_config.weight_init_mean,
                                config.genome_config.weight_init_stdev
                            )
                            conn_rev.enabled = True
                            genome.connections[key_rev] = conn_rev
                            
                            # Add triangle-closing edges with 30% probability
                            # This helps increase clustering coefficient
                            for w in range(self.n_hidden):
                                if w != u and w != v:
                                    # If w is connected to u, connect it to v
                                    key_uw = (int(self.n_in + u), int(self.n_in + w))
                                    key_wv = (int(self.n_in + w), int(self.n_in + v))
                                    if key_uw in genome.connections and np.random.random() < 0.3:
                                        conn_wv = neat.genome.DefaultConnectionGene(key_wv)
                                        conn_wv.weight = np.random.normal(
                                            config.genome_config.weight_init_mean,
                                            config.genome_config.weight_init_stdev
                                        )
                                        conn_wv.enabled = True
                                        genome.connections[key_wv] = conn_wv
                    
                    # Add input connections (2 per input)
                    for i in range(self.n_in):
                        # Choose 2 random hidden nodes for each input
                        hidden_nodes = np.random.choice(self.n_hidden, size=2, replace=False)
                        for h in hidden_nodes:
                            key = (int(i), int(self.n_in + h))  # input -> hidden
                            conn = neat.genome.DefaultConnectionGene(key)
                            conn.weight = np.random.normal(
                                config.genome_config.weight_init_mean,
                                config.genome_config.weight_init_stdev
                            )
                            conn.enabled = True
                            genome.connections[key] = conn
                    
                    # Add output connections (2 per output)
                    for o in range(self.n_out):
                        # Choose 2 random hidden nodes for each output
                        hidden_nodes = np.random.choice(self.n_hidden, size=2, replace=False)
                        for h in hidden_nodes:
                            key = (int(self.n_in + h), int(self.n_in + self.n_hidden + o))  # hidden -> output
                            conn = neat.genome.DefaultConnectionGene(key)
                            conn.weight = np.random.normal(
                                config.genome_config.weight_init_mean,
                                config.genome_config.weight_init_stdev
                            )
                            conn.enabled = True
                            genome.connections[key] = conn
                
                # Adjust deletion probability dynamically
                current_core_density = (
                    sum(1 for c in genome.connections.values() if c.enabled
                        and self.n_in <= c.key[0] < self.n_in+self.n_hidden
                        and self.n_in <= c.key[1] < self.n_in+self.n_hidden)
                    / (self.n_hidden * (self.n_hidden - 1))
                )
                if current_core_density < self.target_density * 0.95:
                    config.genome_config.conn_delete_prob = 0.05  # conservative deletion
                elif current_core_density > self.target_density * 1.05:
                    config.genome_config.conn_delete_prob = 0.20  # aggressive deletion
                else:
                    config.genome_config.conn_delete_prob = 0.10  # normal deletion
                
                # Get the maximum node index from the genome
                max_node = max(
                    max(conn.key[0] for conn in genome.connections.values()),
                    max(conn.key[1] for conn in genome.connections.values()),
                    max(genome.nodes.keys()) if genome.nodes else 0,
                    self.n_total - 1  # Ensure we have enough space for all nodes
                )
                
                # Create adjacency matrix with enough space for all nodes
                size = max(self.n_total, max_node + 1)
                adj_matrix = np.zeros((size, size), dtype=bool)
                
                # Add connections from genome
                for conn in genome.connections.values():
                    if conn.enabled:
                        adj_matrix[conn.key[0]][conn.key[1]] = True
                
                # Convert to torch tensor and wire inputs/outputs
                adj_mask = torch.from_numpy(adj_matrix)
                adj_mask = _wire_inputs_outputs(adj_mask, self.n_in, self.n_hidden, self.n_out, fan_k=2)
                
                # Get core graph (hidden nodes only)
                core = adj_mask[self.n_in:-self.n_out, self.n_in:-self.n_out]
                G_core = nx.from_numpy_array(core.cpu().numpy(), create_using=nx.DiGraph)
                G_core_undirected = G_core.to_undirected()
                
                # Calculate metrics on core graph
                clustering = nx.average_clustering(G_core_undirected)
                path_length = nx.average_shortest_path_length(G_core_undirected)
                
                # Calculate core density
                core_edges = core.sum()
                target_core_edges = int(self.target_density * self.n_hidden * (self.n_hidden - 1))
                core_density = core_edges / (self.n_hidden * (self.n_hidden - 1))
                
                # Check connectivity
                n_components = nx.number_connected_components(G_core_undirected)
                if n_components > 1:
                    genome.fitness = 0.0
                    continue
                
                # Normalized objectives (all in [0,1] range, higher is better)
                C_norm = min(clustering / self.target_clustering, 1.0)  # <=1 is good
                L_norm = min(self.target_path_length / max(path_length, 1e-3), 1.0)  # >=1 is good
                D_norm = max(0, 1 - abs(core_density - self.target_density) / self.target_density)
                
                # Combined fitness with geometric mean and quadratic clustering term
                genome.fitness = float((C_norm**2 * L_norm * D_norm) ** 0.25)
                
                # Early termination if perfect solution
                if (clustering >= self.target_clustering and 
                    abs(core_density - self.target_density) <= 0.01 and
                    abs(path_length - self.target_path_length) <= 0.1 and
                    n_components == 1):
                    genome.fitness = 1.0
                    genome.terminated = True
                    
            except Exception as e:
                print(f"Error evaluating genome {genome_id}: {str(e)}")
                genome.fitness = 0.0

def get_neat_config(n_in: int, n_hidden: int, n_out: int, pop_size: int) -> neat.Config:
    """
    Get NEAT configuration.
    
    Args:
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        pop_size: Population size
        
    Returns:
        NEAT configuration
    """
    # Create config dictionary
    config_dict = {
        'NEAT': {
            'fitness_criterion': 'max',
            'fitness_threshold': 1.0,
            'pop_size': pop_size,
            'reset_on_extinction': False
        },
        'DefaultGenome': {
            'num_inputs': n_in,
            'num_hidden': n_hidden,
            'num_outputs': n_out,
            'initial_connection': 'unconnected',  # Valid values: 'unconnected', 'full', 'partial'
            'feed_forward': True,
            
            # Compatibility coefficients
            'compatibility_disjoint_coefficient': 1.0,
            'compatibility_weight_coefficient': 0.5,
            
            # Node activation options
            'activation_default': 'tanh',
            'activation_options': 'tanh',
            'activation_mutate_rate': 0.1,
            'activation_replace_rate': 0.1,
            
            # Node aggregation options
            'aggregation_default': 'sum',
            'aggregation_options': 'sum',
            'aggregation_mutate_rate': 0.1,
            'aggregation_replace_rate': 0.1,
            
            # Node add/remove rates
            'node_add_prob': 0.2,
            'node_delete_prob': 0.1,
            
            # Connection add/remove rates
            'conn_add_prob': 0.4,  # bias towards adding
            'conn_delete_prob': 0.05,  # initial value; overridden dynamically
            
            # Node bias options
            'bias_init_mean': 0.0,
            'bias_init_stdev': 1.0,
            'bias_max_value': 30.0,
            'bias_min_value': -30.0,
            'bias_replace_rate': 0.1,
            'bias_mutate_rate': 0.7,
            'bias_mutate_power': 0.5,
            
            # Node response options
            'response_init_mean': 1.0,
            'response_init_stdev': 0.0,
            'response_max_value': 30.0,
            'response_min_value': -30.0,
            'response_replace_rate': 0.1,
            'response_mutate_rate': 0.7,
            'response_mutate_power': 0.5,
            
            # Node weight options
            'weight_init_mean': 0.0,
            'weight_init_stdev': 1.0,
            'weight_max_value': 30.0,
            'weight_min_value': -30.0,
            'weight_replace_rate': 0.1,
            'weight_mutate_rate': 0.8,
            'weight_mutate_power': 0.5,
            
            # Connection enable options
            'enabled_default': True,
            'enabled_mutate_rate': 0.01,
            
            # Connection options
            'conn_enabled_default': True,
            'conn_enabled_mutate_rate': 0.01
        },
        'DefaultSpeciesSet': {
            'compatibility_threshold': 3.0
        },
        'DefaultStagnation': {
            'species_fitness_func': 'max',
            'max_stagnation': 20
        },
        'DefaultReproduction': {
            'elitism': 2,
            'survival_threshold': 0.2
        }
    }
    
    # Create temporary config file
    temp_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_neat_config.txt')
    
    # Write config to file
    with open(temp_config_path, 'w') as f:
        for section, params in config_dict.items():
            f.write(f'[{section}]\n')
            for key, value in params.items():
                if isinstance(value, str):
                    f.write(f'{key} = {value}\n')
                else:
                    f.write(f'{key} = {value}\n')
            f.write('\n')
    
    try:
        # Create NEAT config
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_config_path
        )
        return config
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

class FinalWinnerReporter(neat.reporting.BaseReporter):
    """Report final winner statistics."""
    
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        """Initialize reporter with network parameters."""
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
    def post_evaluate(self, config, population, species, best_genome):
        """Called after fitness evaluation."""
        if best_genome is None:
            return
            
        # Convert genome to adjacency matrix
        n_total = self.n_in + self.n_hidden + self.n_out
        adj_matrix = torch.zeros((n_total, n_total), dtype=torch.bool)
        
        # Add connections
        for conn in best_genome.connections.values():
            if conn.enabled:
                # Ensure indices are within bounds
                if 0 <= conn.key[0] < n_total and 0 <= conn.key[1] < n_total:
                    adj_matrix[conn.key[0]][conn.key[1]] = True
        
        # Calculate statistics
        stats = get_network_stats(adj_matrix, self.n_in, self.n_hidden, self.n_out)
        
        # Print statistics
        print("\nFinal network statistics:")
        print(f"Average clustering: {stats['avg_clustering']:.3f}")
        print(f"Average path length: {stats['avg_path_length']}")
        print(f"Network density: {stats['density']:.3f}")
        print(f"Average degree: {stats['avg_degree']:.1f}")
        
        return adj_matrix, stats

def make_sw_neat(
    n_in: int,
    n_hidden: int,
    n_out: int,
    density: float = 0.1,
    target_clustering: float = 0.6,
    target_path_length: float = 2.0,
    pop_size: int = 20,
    n_generations: int = 20,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate a small-world network using NEAT.
    
    Args:
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        density: Target network density
        target_clustering: Target clustering coefficient
        target_path_length: Target average path length
        pop_size: Population size for NEAT
        n_generations: Number of generations to evolve
        seed: Random seed
        
    Returns:
        Tuple of (adjacency matrix, network statistics)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print("\nInitializing NEAT evolution:")
    print(f"- Input nodes: {n_in}")
    print(f"- Hidden nodes: {n_hidden}")
    print(f"- Output nodes: {n_out}")
    print(f"- Population size: {pop_size}")
    print(f"- Generations: {n_generations}")
    print(f"- Target density: {density:.3f}")
    print(f"- Target clustering: {target_clustering:.3f}")
    print(f"- Target path length: {target_path_length:.3f}")
    
    print("\nStarting NEAT evolution...")
    print("-" * 50)
    
    # Create fitness function
    fitness_fn = SmallWorldFitness(
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        target_density=density,
        target_clustering=target_clustering,
        target_path_length=target_path_length
    )
    
    # Get NEAT configuration
    config = get_neat_config(n_in, n_hidden, n_out, pop_size)
    
    # Create population
    pop = neat.Population(config)
    
    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    
    # Add final winner reporter
    final_reporter = FinalWinnerReporter(n_in, n_hidden, n_out)
    pop.add_reporter(final_reporter)
    
    # Run evolution
    best_genome = pop.run(fitness_fn, n_generations)
    
    print("\nEvolution complete!")
    print(f"Final best fitness: {best_genome.fitness:.3f}")
    
    # Get final network
    adj_matrix, stats = final_reporter.post_evaluate(config, pop.population, pop.species, best_genome)
    
    return adj_matrix, stats 