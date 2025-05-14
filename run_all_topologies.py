import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import numpy as np
import random
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from stable_baselines3.common.vec_env import VecMonitor

from topologies.masked_linear import MaskedLinear
from topologies.fully_connected import make_fc
from topologies.random_sparse import make_rs
from topologies.small_world_neat import make_sw_neat
from topologies.watts_strogatz import make_ws
from topologies.modular import make_modular
from topologies.viz import plot_connectivity, plot_network, plot_degree_distribution
from topologies.sb3_integration import create_masked_policy_kwargs, zero_mask_grad

class TrainingCallback(BaseCallback):
    """Custom callback for printing training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.summary = {
            'best_mean_reward': -np.inf,
            'final_mean_reward': 0,
            'final_mean_length': 0,
            'total_episodes': 0,
            'network_stats': {}
        }
        
    def _on_step(self) -> bool:
        # Get episode info if available
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
                    self.episode_count += 1
                    
                    # Calculate statistics
                    mean_reward = np.mean(self.episode_rewards[-100:])
                    mean_length = np.mean(self.episode_lengths[-100:])
                    
                    # Update summary
                    self.summary['final_mean_reward'] = mean_reward
                    self.summary['final_mean_length'] = mean_length
                    self.summary['total_episodes'] = self.episode_count
                    
                    # Print progress less frequently (every 100 episodes)
                    if self.episode_count % 100000 == 0:  # Changed from 10 to 100
                        print(f"\nEpisode {self.episode_count}")
                        print(f"Mean reward (last 100000): {mean_reward:.1f}")
                        print(f"Mean length (last 100000): {mean_length:.1f}")
                        
                        # Print network statistics
                        for module in self.model.policy.modules():
                            if isinstance(module, MaskedLinear):
                                stats = module.get_structural_stats()
                                print("\nNetwork Statistics:")
                                print(f"- Sparsity: {stats['sparsity']:.3f}")
                                print(f"- Avg Clustering: {stats['avg_clustering']:.3f}")
                                print(f"- Avg Degree: {stats['avg_degree']:.1f}")
                                if 'avg_path_length' in stats:
                                    print(f"- Avg Path Length: {stats['avg_path_length']:.2f}")
                                
                                # Update network stats in summary
                                self.summary['network_stats'] = stats
                    
                    # Save best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.summary['best_mean_reward'] = mean_reward
                        self.model.save(os.path.join(self.model.tensorboard_log, "best_model"))
                        print(f"\nNew best model saved! Mean reward: {mean_reward:.1f}")
        
        return True

class WeightClampingCallback(BaseCallback):
    """Callback to clamp weights after each update."""
    
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> bool:
        self.model.policy.apply(zero_mask_grad)
        return True

def to_python_types(d):
    """Recursively convert all values to Python built-in types."""
    if isinstance(d, dict):
        return {k: to_python_types(v) for k, v in d.items()}
    elif isinstance(d, (np.integer, np.floating)):
        return d.item()
    elif isinstance(d, torch.Tensor):
        return d.item() if d.numel() == 1 else d.tolist()
    else:
        return d

def train_topology(cfg: DictConfig, topology_type: str, run_dir: str) -> None:
    """Train a single topology and save results."""
    print(f"\n=== Training {topology_type} Topology ===")
    
    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Create environment
    base_env = gym.make(cfg.env.name)
    base_env.reset(seed=cfg.seed)
    env = DummyVecEnv([lambda: base_env])
    env = VecMonitor(env)  # Add VecMonitor wrapper to track episode information
    
    # Get input and output dimensions
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    # Create topology mask
    if topology_type == "fc":
        adj_mask, meta = make_fc(n_in, cfg.topology.n_hidden, n_out)
    elif topology_type == "rs":
        adj_mask, meta = make_rs(n_in, cfg.topology.n_hidden, n_out, cfg.topology.density, cfg.seed)
    elif topology_type == "sw_neat":
        adj_mask, meta = make_sw_neat(
            n_in, cfg.topology.n_hidden, n_out,
            density=cfg.topology.density,
            target_clustering=0.6,
            target_path_length=2.0,
            seed=cfg.seed
        )
    elif topology_type == "sw_ws":
        adj_mask, meta = make_ws(n_in, cfg.topology.n_hidden, n_out)
    elif topology_type == "modular":
        adj_mask, meta = make_modular(
            n_in, cfg.topology.n_hidden, n_out,
            n_modules=cfg.topology.n_modules,
            p_intra=cfg.topology.p_intra,
            p_inter=cfg.topology.p_inter,
            seed=cfg.seed
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    # Create topology directory
    topology_dir = os.path.join(run_dir, topology_type)
    os.makedirs(topology_dir, exist_ok=True)
    
    # Save topology metadata
    with open(os.path.join(topology_dir, "topology_meta.yaml"), "w") as f:
        yaml.dump(to_python_types(meta), f)
    
    # Create policy kwargs with masked topology
    policy_kwargs = create_masked_policy_kwargs(
        adj_mask=adj_mask,
        n_in=n_in,
        n_out=n_out,
        features_dim=cfg.topology.n_hidden,
        hidden_act=torch.nn.ReLU(),
        out_act=torch.nn.Identity()
    )
    
    # Create callbacks
    training_callback = TrainingCallback()
    callbacks = [
        training_callback,
        WeightClampingCallback()
    ]
    
    # Create and train agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.training.lr,
        n_steps=cfg.training.n_steps,
        batch_size=cfg.training.batch_size,
        n_epochs=cfg.training.n_epochs,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_range=cfg.training.clip_range,
        verbose=1,
        tensorboard_log=topology_dir
    )
    
    # Train
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        tb_log_name=topology_type
    )
    
    # Save final model
    model.save(os.path.join(topology_dir, "final_model"))
    
    # Save visualizations
    plot_connectivity(
        adj_mask,
        f"{topology_type.upper()} Topology",
        os.path.join(topology_dir, "connectivity.png")
    )
    plot_network(
        adj_mask,
        f"{topology_type.upper()} Network",
        os.path.join(topology_dir, "network.png")
    )
    plot_degree_distribution(
        adj_mask,
        f"{topology_type.upper()} Degree Distribution",
        os.path.join(topology_dir, "degree_dist.png")
    )
    
    # Print training summary
    print(f"\n=== Training Summary for {topology_type.upper()} ===")
    print(f"Total Episodes: {training_callback.summary['total_episodes']}")
    print(f"Best Mean Reward: {training_callback.summary['best_mean_reward']:.2f}")
    print(f"Final Mean Reward: {training_callback.summary['final_mean_reward']:.2f}")
    print(f"Final Mean Episode Length: {training_callback.summary['final_mean_length']:.2f}")
    
    if training_callback.summary['network_stats']:
        print("\nNetwork Statistics:")
        stats = training_callback.summary['network_stats']
        print(f"- Sparsity: {stats['sparsity']:.3f}")
        print(f"- Avg Clustering: {stats['avg_clustering']:.3f}")
        print(f"- Avg Degree: {stats['avg_degree']:.1f}")
        if 'avg_path_length' in stats:
            print(f"- Avg Path Length: {stats['avg_path_length']:.2f}")
    
    print(f"\nResults saved to: {topology_dir}")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run training for all topologies."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"topology_comparison_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    # Define topologies to run
    topologies = [
        "fc",           # Fully Connected
        "modular",      # Modular
        "rs",           # Random Sparse
        "sw_neat",      # Small World NEAT
        "sw_ws"         # Small World Watts-Strogatz
    ]
    
    # Run training for each topology
    for topology in topologies:
        train_topology(cfg, topology, run_dir)
    
    print(f"\nAll training complete! Results saved to: {run_dir}")

if __name__ == "__main__":
    main() 