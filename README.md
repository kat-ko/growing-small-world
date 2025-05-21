# Growing Small World Networks

## Results Folder Structure

When running `run_all_topologies.py`, the following folder structure is generated in the `results` directory:

```
results/
├── {timestamp}/                      # Each run creates a timestamped folder
│   ├── config.yaml                   # Copy of configuration used for this run
│   ├── topology_manifest.json        # Metadata about all generated topologies
│   ├── {topology_type}/             # One folder per topology type (fc, rs, sw_neat, modular)
│   │   ├── seed_{seed_num}/         # One folder per random seed
│   │   │   ├── model/               # Saved model checkpoints
│   │   │   │   ├── best_model.zip   # Best performing model
│   │   │   │   └── final_model.zip  # Final model state
│   │   │   ├── metrics/             # Training metrics
│   │   │   │   ├── rewards.csv      # Reward history
│   │   │   │   ├── losses.csv       # Loss history
│   │   │   │   └── metrics.json     # Summary metrics
│   │   │   ├── network/             # Network visualization
│   │   │   │   ├── network.png      # Network graph visualization
│   │   │   │   └── network_stats.json # Network statistics
│   │   │   └── training/            # Training logs
│   │   │       ├── events.out.tfevents.* # TensorBoard logs
│   │   │       └── training.log     # Training console output
│   │   └── summary/                 # Aggregated results across seeds
│   │       ├── mean_rewards.png     # Plot of mean rewards
│   │       ├── std_rewards.png      # Plot of reward standard deviations
│   │       └── summary.json         # Statistical summary
│   └── comparison/                  # Cross-topology comparisons
│       ├── reward_comparison.png    # Reward comparison plot
│       ├── network_stats_comparison.png # Network statistics comparison
│       └── comparison.json          # Comparison metrics
```

### File Generation Process

```python
# Pseudo-code for file generation
def run_all_topologies():
    timestamp = get_current_timestamp()
    results_dir = f"results/{timestamp}"
    
    # Copy config
    copy_config("config/config.yaml", f"{results_dir}/config.yaml")
    
    # Generate and save topology manifest
    manifest = generate_topology_manifest()
    save_json(manifest, f"{results_dir}/topology_manifest.json")
    
    for topology_type in ["fc", "rs", "sw_neat", "modular"]:
        topology_dir = f"{results_dir}/{topology_type}"
        
        for seed in seeds:
            seed_dir = f"{topology_dir}/seed_{seed}"
            
            # Train model
            model = train_model(topology_type, seed)
            
            # Save model checkpoints
            save_model(model.best_model, f"{seed_dir}/model/best_model.zip")
            save_model(model.final_model, f"{seed_dir}/model/final_model.zip")
            
            # Save metrics
            save_csv(model.rewards, f"{seed_dir}/metrics/rewards.csv")
            save_csv(model.losses, f"{seed_dir}/metrics/losses.csv")
            save_json(model.metrics, f"{seed_dir}/metrics/metrics.json")
            
            # Generate and save network visualization
            network_graph = visualize_network(model.network)
            save_image(network_graph, f"{seed_dir}/network/network.png")
            save_json(model.network_stats, f"{seed_dir}/network/network_stats.json")
            
            # Save training logs
            save_tensorboard_logs(model.logs, f"{seed_dir}/training/events.out.tfevents.*")
            save_log(model.console_output, f"{seed_dir}/training/training.log")
        
        # Generate summary across seeds
        generate_summary(topology_dir)
    
    # Generate cross-topology comparisons
    generate_comparisons(results_dir)
```

## Relationship to Project Objectives

The results folder structure directly supports the project's main objectives:

1. **Topology Comparison**
   - `comparison/` folder contains cross-topology analyses
   - `reward_comparison.png` shows performance differences
   - `network_stats_comparison.png` shows structural differences
   - `comparison.json` provides quantitative metrics

2. **Network Analysis**
   - `network/` subfolders contain detailed network visualizations
   - `network_stats.json` files track key metrics:
     - Sparsity
     - Clustering coefficient
     - Average path length
     - Degree distribution

3. **Training Performance**
   - `metrics/` folders track training progress
   - `rewards.csv` shows learning curves
   - `losses.csv` tracks optimization progress
   - `metrics.json` contains final performance metrics

4. **Reproducibility**
   - `config.yaml` ensures experiment reproducibility
   - `topology_manifest.json` documents network generation
   - Multiple seeds enable statistical significance
   - Timestamped folders prevent overwriting

5. **Model Persistence**
   - `model/` folders store trained models
   - `best_model.zip` contains best performing model
   - `final_model.zip` contains final model state

6. **Visualization and Analysis**
   - Network visualizations in `network.png`
   - Training curves in `mean_rewards.png`
   - Performance distributions in `std_rewards.png`
   - TensorBoard logs for detailed analysis

This structure enables:
- Systematic comparison of different network topologies
- Detailed analysis of network properties
- Performance evaluation across multiple seeds
- Easy reproduction of experiments
- Comprehensive visualization of results 