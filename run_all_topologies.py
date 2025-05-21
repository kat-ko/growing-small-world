import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import numpy as np
import random
import yaml
from pathlib import Path
import pytest
from tqdm.auto import tqdm

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
from topologies.sb3_integration import create_masked_policy_kwargs, zero_mask_grad, WeightClampingCallback
from topologies.utils import io_reachability

class TrainingCallback(BaseCallback):
    """Custom callback for printing training progress using tqdm."""
    
    def __init__(self, total_timesteps: int, description: str, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.description = description
        self.pbar = None
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
        
    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        self.pbar = tqdm(total=self.total_timesteps, desc=self.description, unit="step")
        
    def _on_step(self) -> bool:
        # Update TQDM progress bar
        if self.pbar:
            self.pbar.update(1)

        # Get episode info if available
        if self.locals['infos']: 
            info = self.locals['infos'][0]
            if 'episode' in info:  
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                self.episode_count += 1
                
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                self.summary['final_mean_reward'] = mean_reward
                self.summary['final_mean_length'] = mean_length
                self.summary['total_episodes'] = self.episode_count
                
                # Update TQDM postfix
                if self.pbar:
                    self.pbar.set_postfix(
                        ep_rew_mean=f"{mean_reward:.2f}", 
                        ep_len_mean=f"{mean_length:.2f}",
                        episodes=self.episode_count
                    )
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.summary['best_mean_reward'] = mean_reward
                    if self.model.tensorboard_log:
                         self.model.save(os.path.join(self.model.tensorboard_log, "best_model"))
        
        return True

    def _on_rollout_end(self) -> bool:
        """Called at the end of each rollout collection."""
        if self.model and self.model.policy and hasattr(self.model.policy, 'features_extractor'):
            extractor = self.model.policy.features_extractor
            # Find the MaskedLinear layer, similar to the test
            mlayer = next((m for m in extractor.modules() if isinstance(m, MaskedLinear)), None)
            if mlayer:
                try:
                    stats = mlayer.get_structural_stats()
                    self.summary['network_stats'] = stats
                except Exception as e:
                    print(f"Warning: Could not retrieve structural stats in TrainingCallback: {e}")
            # else: # Optional: print a warning if no MaskedLinear layer is found
            #     print("Warning: No MaskedLinear layer found in features_extractor for stats collection.")
        return True

    def _on_training_end(self) -> None:
        """Called once the training ended."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

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

def active_features(mask, n_in, n_hidden, n_out):
    # count hidden columns that receive at least one connection from any input row
    inp_rows = mask[:n_in, n_in:-n_out]
    return (inp_rows.sum(dim=0) > 0).sum().item()

def train_topology(cfg: DictConfig, topology_type: str, run_dir: str, target_n_weights: int) -> dict:
    """Train a single topology and save results."""
    print(f"\n=== Training {topology_type} Topology ===")
    
    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Create topology directory (used for monitor.csv and other outputs)
    topology_specific_output_dir = os.path.join(run_dir, topology_type)
    os.makedirs(topology_specific_output_dir, exist_ok=True)
    monitor_file_path = os.path.join(topology_specific_output_dir, "monitor.csv")
    
    # Create environment
    base_env = gym.make(cfg.env.name)
    base_env.reset(seed=cfg.seed)
    env = DummyVecEnv([lambda: base_env])
    env = VecMonitor(env, filename=monitor_file_path)
    
    # Get input and output dimensions
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    # Create topology mask
    if topology_type == "fc":
        adj_mask, meta = make_fc(n_in, cfg.topology.n_hidden, n_out)
    elif topology_type == "rs":
        adj_mask, meta = make_rs(n_in, cfg.topology.n_hidden, n_out, 
                               density=cfg.topology.density, 
                               seed=cfg.seed,
                               target_n_weights=target_n_weights)
    elif topology_type == "sw_neat":
        adj_mask, meta = make_sw_neat(
            n_in, cfg.topology.n_hidden, n_out,
            density=cfg.topology.density,
            target_clustering=0.6,
            target_path_length=2.0,
            seed=cfg.seed,
            target_n_weights=target_n_weights
        )
    elif topology_type == "sw_ws":
        adj_mask, meta = make_ws(n_in, cfg.topology.n_hidden, n_out,
                               k=cfg.topology.k,
                               p=cfg.topology.p,
                               seed=cfg.seed,
                               target_n_weights=target_n_weights)
    elif topology_type == "modular":
        adj_mask, meta = make_modular(
            n_in, cfg.topology.n_hidden, n_out,
            n_modules=cfg.topology.n_modules,
            p_intra=cfg.topology.p_intra,
            p_inter=cfg.topology.p_inter,
            seed=cfg.seed,
            target_n_weights=target_n_weights
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    # Calculate and store active hidden units
    meta['active_hidden'] = active_features(adj_mask, n_in, cfg.topology.n_hidden, n_out)
    print(f"Active hidden units: {meta['active_hidden']} / {cfg.topology.n_hidden}")
    
    # Calculate and store IO reachability
    if isinstance(adj_mask, torch.Tensor): # Ensure adj_mask is a tensor before passing
        meta['reachability'] = io_reachability(adj_mask, n_in, cfg.topology.n_hidden, n_out)
        print(f"IO reachability for {topology_type}: {meta['reachability']:.3f}") # Print with 3 decimal places
    else:
        # This case should ideally not be hit if make_* functions are consistent
        print(f"Warning: adj_mask for {topology_type} is not a Tensor. Skipping reachability calculation.")
        meta['reachability'] = -1.0 # Or some other indicator for missing data

    # Save topology metadata (meta now includes reachability)
    with open(os.path.join(topology_specific_output_dir, "topology_meta.yaml"), "w") as f:
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
    
    # Add net_arch to policy_kwargs
    policy_kwargs["net_arch"] = []
    
    # Create callbacks
    progress_bar_desc = f"Training {topology_type.upper()} (Seed {cfg.seed})"
    training_callback = TrainingCallback(
        total_timesteps=cfg.training.total_timesteps,
        description=progress_bar_desc
    )
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
        verbose=0,
        tensorboard_log=topology_specific_output_dir # <--- USE new path (SB3 will create tb_log_name subdir in here)
    )
    
    # Train
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        tb_log_name=topology_type
    )
    
    # Save final model
    model.save(os.path.join(topology_specific_output_dir, "final_model"))
    
    # Save visualizations
    plot_connectivity(
        adj_mask,
        f"{topology_type.upper()} Topology",
        os.path.join(topology_specific_output_dir, "connectivity.png")
    )
    plot_network(
        adj_mask,
        f"{topology_type.upper()} Network",
        os.path.join(topology_specific_output_dir, "network.png")
    )
    plot_degree_distribution(
        adj_mask,
        f"{topology_type.upper()} Degree Distribution",
        os.path.join(topology_specific_output_dir, "degree_dist.png")
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
    
    print(f"\nResults saved to: {topology_specific_output_dir}")

    return training_callback.summary

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run training for all topologies across multiple seeds."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"topology_comparison_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save main config for the entire run
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

    # Compute baseline weight budget once
    env = gym.make(cfg.env.name)
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    n_w_fc = (n_in + cfg.topology.n_hidden) * cfg.topology.n_hidden + \
             (cfg.topology.n_hidden + 1) * n_out  # +1 for biases
    target_n_weights = int(n_w_fc)
    print(f"\nTarget weight budget: {target_n_weights}")
    env.close()

    n_seeds = cfg.evaluation.get('n_seeds', 1)
    print(f"\n=== Starting Experiment Run for {n_seeds} Seed(s) ===")

    for seed_idx in range(n_seeds):
        current_seed = cfg.seed + seed_idx
        cfg_copy = cfg.copy()
        cfg_copy.seed = current_seed
        
        print(f"\n--- Running Seed {seed_idx + 1}/{n_seeds} (Actual Seed Value: {current_seed}) ---")
        seed_dir = os.path.join(run_dir, f"seed_{current_seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_summaries = {}
        for topology_type in topologies:
            summary = train_topology(cfg_copy, topology_type, seed_dir, target_n_weights)
            seed_summaries[topology_type] = summary

        # save one YAML per seed
        seed_summary_file = os.path.join(seed_dir, "training_summaries.yaml")
        with open(seed_summary_file, "w") as f:
            yaml.dump(to_python_types(seed_summaries), f, indent=4)
        print(f"Training summaries for seed {current_seed} saved to: {seed_summary_file}")
    
    print(f"\nAll training across all seeds complete! Results saved to: {run_dir}")

    # Run pytest for unit tests
    print("\n=== Running Unit Tests ===")
    test_exit_code = pytest.main(["-q", "tests/test_mask.py"])
    if test_exit_code == 0:
        print("All tests passed!")
    else:
        print(f"Pytest finished with exit code: {test_exit_code}. Some tests may have failed.")

if __name__ == "__main__":
    main() 