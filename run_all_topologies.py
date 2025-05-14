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

def train_topology(cfg: DictConfig, topology_type: str, run_dir: str) -> dict:
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

    # --- new outer loop over seeds ---
    # Ensure cfg.evaluation.n_seeds exists in your Hydra config
    # For example, in config.yaml: 
    # evaluation:
    #   n_seeds: 3
    n_seeds = cfg.evaluation.get('n_seeds', 1) # Default to 1 seed if not specified
    print(f"\n=== Starting Experiment Run for {n_seeds} Seed(s) ===")

    for seed_idx in range(n_seeds):
        current_seed = cfg.seed + seed_idx # Allow overriding initial seed from CLI, then increment
                                         # Or, if you prefer strict seed numbers 0,1,2... use seed_idx directly
                                         # For now, using initial cfg.seed as base and incrementing.
                                         # If cfg.seed is 0, seeds will be 0, 1, 2... for n_seeds=3.
                                         # If cfg.seed is 42, seeds will be 42, 43, 44...
        # Update config with the current seed for this iteration
        cfg_copy = cfg.copy() # Operate on a copy for this seed run to avoid polluting original cfg for next seed
        cfg_copy.seed = current_seed 
        
        print(f"\n--- Running Seed {seed_idx + 1}/{n_seeds} (Actual Seed Value: {current_seed}) ---")
        seed_dir = os.path.join(run_dir, f"seed_{current_seed}") # Use actual seed value in dir name
        os.makedirs(seed_dir, exist_ok=True)

        seed_summaries = {}                 # collect summaries for this seed
        for topology_type in topologies:         # existing inner loop (renamed topology to topology_type for clarity)
            summary = train_topology(cfg_copy, topology_type, seed_dir) # Pass cfg_copy and seed_dir
            seed_summaries[topology_type] = summary

        # save one YAML per seed
        seed_summary_file = os.path.join(seed_dir, "training_summaries.yaml")
        with open(seed_summary_file, "w") as f:
            yaml.dump(to_python_types(seed_summaries), f, indent=4)
        print(f"Training summaries for seed {current_seed} saved to: {seed_summary_file}")
    
    print(f"\nAll training across all seeds complete! Results saved to: {run_dir}")

    # Run pytest for unit tests (outside the seed loop, runs once)
    print("\n=== Running Unit Tests ===")
    # test_exit_code = pytest.main(["-v", "tests/test_mask.py"]) # Use -q for less verbose output as requested
    test_exit_code = pytest.main(["-q", "tests/test_mask.py"])
    if test_exit_code == 0:
        print("All tests passed!")
    else:
        print(f"Pytest finished with exit code: {test_exit_code}. Some tests may have failed.")

if __name__ == "__main__":
    main() 