import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from datetime import datetime
from typing import Any
import numpy as np
import random
from stable_baselines3.common.callbacks import BaseCallback

from topologies.masked_linear import MaskedLinear
from topologies.fully_connected import make_fc
from topologies.random_sparse import make_rs
from topologies.small_world_neat import make_sw_neat
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
                    
                    # Print progress
                    if self.episode_count % 10 == 0:  # Print every 10 episodes
                        print(f"\nEpisode {self.episode_count}")
                        print(f"Mean reward (last 100): {mean_reward:.1f}")
                        print(f"Mean length (last 100): {mean_length:.1f}")
                        
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
                    
                    # Save best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.model.save(os.path.join(self.model.tensorboard_log, "best_model"))
                        print(f"\nNew best model saved! Mean reward: {mean_reward:.1f}")
        
        return True

class WeightClampingCallback(BaseCallback):
    """Callback to clamp weights after each update."""
    
    def _on_step(self) -> bool:
        self.model.policy.apply(zero_mask_grad)
        return True

def to_python_types(d: Any):
    """Recursively convert all values to Python built-in types."""
    if isinstance(d, dict):
        return {k: to_python_types(v) for k, v in d.items()}
    elif isinstance(d, (np.integer, np.floating)):
        return d.item()
    elif isinstance(d, torch.Tensor):
        return d.item() if d.numel() == 1 else d.tolist()
    else:
        return d

@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    
    print("\n=== Training Configuration ===")
    print(f"Environment: {cfg.env.name}")
    print(f"Topology: {cfg.topology.type}")
    print(f"Hidden units: {cfg.topology.n_hidden}")
    print(f"Target density: {cfg.topology.density}")
    print(f"Learning rate: {cfg.training.lr}")
    print(f"Total timesteps: {cfg.training.total_timesteps}")
    print(f"Random seed: {cfg.seed}")
    print("===========================\n")
    
    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{cfg.env.name}_{cfg.topology.type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    # Create environment
    env = gym.make(cfg.env.name)
    env.reset(seed=cfg.seed)  # Seed the environment
    env = DummyVecEnv([lambda: env])
    
    # Get input and output dimensions
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    print("\n=== Creating Network Topology ===")
    # Create topology mask
    if cfg.topology.type == "fc":
        adj_mask, meta = make_fc(n_in, cfg.topology.n_hidden, n_out)
    elif cfg.topology.type == "rs":
        adj_mask, meta = make_rs(n_in, cfg.topology.n_hidden, n_out, cfg.topology.density, cfg.seed)
    elif cfg.topology.type == "sw":
        adj_mask, meta = make_sw_neat(n_in, cfg.topology.n_hidden, n_out, cfg.topology.density, cfg.seed)
    elif cfg.topology.type == "modular":
        adj_mask, meta = make_modular(
            n_in, cfg.topology.n_hidden, n_out,
            n_modules=cfg.topology.n_modules,
            p_intra=cfg.topology.p_intra,
            p_inter=cfg.topology.p_inter,
            seed=cfg.seed
        )
    else:
        raise ValueError(f"Unknown topology type: {cfg.topology.type}")
    
    print("\n=== Network Statistics ===")
    print(f"Input nodes: {n_in}")
    print(f"Hidden nodes: {cfg.topology.n_hidden}")
    print(f"Output nodes: {n_out}")
    
    # Print available metrics
    if 'density' in meta:
        print(f"Network density: {meta['density']:.3f}")
    if 'sparsity' in meta:
        print(f"Network sparsity: {meta['sparsity']:.3f}")
    if 'avg_clustering' in meta:
        print(f"Average clustering: {meta['avg_clustering']:.3f}")
    if 'avg_path_length' in meta:
        print(f"Average path length: {meta['avg_path_length']:.2f}")
    if 'avg_degree' in meta:
        print(f"Average degree: {meta['avg_degree']:.1f}")
    print("========================\n")
    
    # Create policy kwargs with masked topology
    policy_kwargs = create_masked_policy_kwargs(
        adj_mask=adj_mask,
        n_in=n_in,
        n_out=n_out,
        features_dim=cfg.topology.n_hidden,  # Must match n_hidden for proper feature extraction
        hidden_act=torch.nn.ReLU(),
        out_act=torch.nn.Identity()
    )
    
    print("\n=== Initializing PPO ===")
    # Create and train agent
    model = PPO(
        policy="MlpPolicy",  # Using MlpPolicy since we're replacing the feature extractor
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
        tensorboard_log=run_dir
    )
    
    # Create callbacks
    callbacks = [
        TrainingCallback(),  # Custom callback for progress printing
        WeightClampingCallback()  # Weight clamping callback
    ]
    
    print("\n=== Starting Training ===")
    # Train
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        tb_log_name=f"{cfg.topology.type}"
    )
    
    print("\n=== Training Complete ===")
    # Verify masked weights are zero
    for module in model.policy.modules():
        if isinstance(module, MaskedLinear):
            with torch.no_grad():
                # Check that masked weights are zero
                masked_weights = module.W * (1 - module.mask)
                assert torch.allclose(masked_weights, torch.zeros_like(masked_weights)), \
                    "Some masked weights are non-zero!"
                # Check that masked biases are zero
                if module.bias is not None:
                    masked_biases = module.bias[module.mask.sum(0) == 0]
                    assert torch.allclose(masked_biases, torch.zeros_like(masked_biases)), \
                        "Some masked biases are non-zero!"
                print("✓ Mask integrity check passed!")
    
    # Save model
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Model saved to {run_dir}")
    
    # Save topology metadata
    with open(os.path.join(run_dir, "topology_meta.yaml"), "w") as f:
        OmegaConf.save(to_python_types(meta), f)
    
    # Save topology visualizations (after training)
    if cfg.viz.save_plots:
        print("\n=== Saving Visualizations ===")
        plot_connectivity(
            adj_mask,
            f"{cfg.topology.type.upper()} Topology",
            os.path.join(run_dir, "connectivity.png")
        )
        plot_network(
            adj_mask,
            f"{cfg.topology.type.upper()} Network",
            os.path.join(run_dir, "network.png")
        )
        plot_degree_distribution(
            adj_mask,
            f"{cfg.topology.type.upper()} Degree Distribution",
            os.path.join(run_dir, "degree_dist.png")
        )
        print("Visualizations saved!")

if __name__ == "__main__":
    train() 