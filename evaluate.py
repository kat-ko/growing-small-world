import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json

from topologies.masked_linear import MaskedLinear
from topologies.fully_connected import make_fc
from topologies.random_sparse import make_rs
from topologies.small_world_neat import make_sw_neat
from topologies.modular import make_modular
from topologies.viz import plot_connectivity, plot_network, plot_degree_distribution

def run_experiment(
    env_name: str,
    topology_type: str,
    n_hidden: int,
    density: float = 0.1,
    n_modules: int = 4,
    p_intra: float = 0.8,
    p_inter: float = 0.1,
    seed: int = 42,
    total_timesteps: int = 1000000,
    eval_episodes: int = 100
) -> Dict[str, Any]:
    """Run a single experiment with given parameters."""
    
    # Create environment
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])
    
    # Get input and output dimensions
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    
    # Create topology mask
    if topology_type == "fc":
        adj_mask, meta = make_fc(n_in, n_hidden, n_out)
    elif topology_type == "rs":
        adj_mask, meta = make_rs(n_in, n_hidden, n_out, density, seed)
    elif topology_type == "sw":
        adj_mask, meta = make_sw_neat(n_in, n_hidden, n_out, seed=seed)
    elif topology_type == "modular":
        adj_mask, meta = make_modular(n_in, n_hidden, n_out, n_modules, p_intra, p_inter, seed)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    # Create policy network
    policy = MaskedLinear(
        adj_mask,
        n_in=n_in,
        n_out=n_out,
        hidden_act=torch.nn.ReLU(),
        out_act=torch.nn.Identity()
    )
    
    # Create and train agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Evaluate
    rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    
    # Compute statistics
    results = {
        "topology": topology_type,
        "n_hidden": n_hidden,
        "density": density if topology_type == "rs" else None,
        "n_modules": n_modules if topology_type == "modular" else None,
        "p_intra": p_intra if topology_type == "modular" else None,
        "p_inter": p_inter if topology_type == "modular" else None,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "sparsity": meta["sparsity"],
        "avg_clustering": meta["avg_clustering"],
        "avg_degree": meta["avg_degree"],
        "seed": seed
    }
    
    if "avg_path_length" in meta:
        results["avg_path_length"] = meta["avg_path_length"]
    if "modularity" in meta:
        results["modularity"] = meta["modularity"]
    
    return results

@hydra.main(config_path="config", config_name="config")
def evaluate(cfg: DictConfig):
    """Run evaluation experiments for different topologies."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"evaluation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    
    # Run experiments
    results = []
    for topology in cfg.evaluation.topologies:
        for seed in range(cfg.evaluation.n_seeds):
            print(f"\nRunning {topology} with seed {seed}")
            result = run_experiment(
                env_name=cfg.env.name,
                topology_type=topology,
                n_hidden=cfg.topology.n_hidden,
                density=cfg.topology.density,
                n_modules=cfg.topology.n_modules,
                p_intra=cfg.topology.p_intra,
                p_inter=cfg.topology.p_inter,
                seed=seed,
                total_timesteps=cfg.training.total_timesteps,
                eval_episodes=cfg.evaluation.n_episodes
            )
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    
    # Create plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="topology", y="mean_reward")
    plt.title("Mean Reward by Topology")
    plt.savefig(os.path.join(results_dir, "reward_boxplot.png"))
    plt.close()
    
    # Plot structural metrics
    metrics = ["sparsity", "avg_clustering", "avg_degree"]
    if "avg_path_length" in df.columns:
        metrics.append("avg_path_length")
    if "modularity" in df.columns:
        metrics.append("modularity")
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="topology", y=metric)
        plt.title(f"{metric.replace('_', ' ').title()} by Topology")
        plt.savefig(os.path.join(results_dir, f"{metric}_boxplot.png"))
        plt.close()
    
    # Print summary
    print("\nResults Summary:")
    print(df.groupby("topology")["mean_reward"].agg(["mean", "std"]))
    
    # Save summary
    summary = {
        "reward_summary": df.groupby("topology")["mean_reward"].agg(["mean", "std"]).to_dict(),
        "structural_summary": {
            metric: df.groupby("topology")[metric].agg(["mean", "std"]).to_dict()
            for metric in metrics
        }
    }
    
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    evaluate() 