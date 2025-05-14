#!/usr/bin/env python
# aggregate_results.py
"""
Collect monitor.csv and training_summaries.yaml from a run folder
and write a tidy parquet that you can inspect or plot quickly.


python aggregate_results.py --results_dir results/topology_comparison_20250514_142447 --out pilot_cartpole.parquet

"""

import argparse, pathlib, yaml, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def monitor_metrics(mfile: pathlib.Path):
    try:
        # Attempt to read, forcing errors on bad lines to be NaNs which can be dropped or handled.
        df = pd.read_csv(mfile, comment="#", names=["r", "l", "t"], header=None) # header=None since we provide names
        
        # Convert to numeric, coercing errors to NaN
        df["r"] = pd.to_numeric(df["r"], errors='coerce')
        df["l"] = pd.to_numeric(df["l"], errors='coerce')
        df["t"] = pd.to_numeric(df["t"], errors='coerce')

        # Drop rows where essential data (reward) could not be parsed
        df.dropna(subset=["r"], inplace=True)

        if df.empty:
            print(f"Warning: No valid numeric data found in {mfile} after parsing.")
            return None                 # unfinished run or all data bad
            
    except pd.errors.ParserError:
        print(f"Warning: Pandas ParserError for file {mfile}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: Could not read or parse {mfile} due to {e}. Skipping.")
        return None

    if df.empty: # Double check after potential drops
        return None

    steps  = df["t"].cumsum()
    reward = df["r"]
    
    if reward.empty: # If all rewards were NaN and dropped
        return None

    final  = reward.tail(100).mean()
    # Use .to_numpy() for np.trapz to ensure compatibility and avoid warnings
    auc    = np.trapezoid(reward.to_numpy(), steps.to_numpy()) / steps.iloc[-1] if not steps.empty and steps.iloc[-1] != 0 else np.nan
    
    try: 
        # Ensure reward.cummax() is not empty before trying to access .iloc[0]
        cummax_reward = reward.cummax()
        if not cummax_reward.empty:
            valid_steps = steps[cummax_reward >= 400]
            step80 = valid_steps.iloc[0] if not valid_steps.empty else np.nan
        else:
            step80 = np.nan
    except IndexError: 
        step80 = np.nan
    except Exception as e: # Catch other potential errors during step80 calculation
        print(f"Warning: Error calculating step80 for {mfile}: {e}")
        step80 = np.nan
        
    return final, auc, step80

def walk_run_dir(root: pathlib.Path):
    rows = []
    print(f"Searching for monitor.csv files in: {root.resolve()}") # DEBUG PRINT
    found_monitors = list(root.rglob("monitor.csv")) # DEBUG: Collect first
    if not found_monitors:
        print("DEBUG: No monitor.csv files found by rglob.") # DEBUG PRINT
    else:
        print(f"DEBUG: Found {len(found_monitors)} monitor.csv files:") # DEBUG PRINT
        for f_path in found_monitors:
            print(f"  - {f_path}") # DEBUG PRINT

    for monitor in found_monitors: # Iterate over the collected list
        topo  = monitor.parent.name          # fc / rs / ...
        seed  = monitor.parents[1].name      # seed_0 / seed_1
        metrics = monitor_metrics(monitor)
        if metrics is None: continue
        final, auc, step80 = metrics
        # read topology stats (optional)
        meta_file = monitor.parent / "topology_meta.yaml"
        meta = yaml.safe_load(meta_file.read_text()) if meta_file.exists() else {}
        rows.append(dict(topology=topo, seed=seed, final=final, auc=auc,
                         step80=step80, **meta))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="results.parquet")
    args = ap.parse_args()
    
    results_path = pathlib.Path(args.results_dir)
    df = walk_run_dir(results_path)
    
    if df.empty:
        print("Warning: No data found. Output Parquet file will be empty or not created.")
    else:
        # Save the main data to Parquet
        df.to_parquet(args.out)
        print(f"Wrote {len(df)} rows to {args.out}")

        # Create a subdirectory for plots within the processed results_dir
        plots_dir = results_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        run_name = results_path.name # Get the name of the specific run folder for titles

        # Generate and save boxplot for final returns
        try:
            plt.figure(figsize=(12, 7)) # Adjusted figure size for potentially more info
            sns.boxplot(x="topology", y="final", data=df)
            plt.title(f"Final Return Distribution\nRun: {run_name}")
            plt.xticks(rotation=45, ha='right') 
            plt.tight_layout() 
            plot_path = plots_dir / "final_return_boxplot.png"
            plt.savefig(plot_path)
            plt.close() 
            print(f"Saved boxplot for final returns to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate or save boxplot for final returns: {e}")

        # New: Generate and save boxplot for reachability
        if 'reachability' in df.columns:
            try:
                plt.figure(figsize=(12, 7))
                sns.boxplot(x="topology", y="reachability", data=df)
                plt.title(f"IO Reachability Distribution\nRun: {run_name}")
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Reachability Score") # Add y-axis label
                plt.tight_layout()
                reach_plot_path = plots_dir / "reachability_boxplot.png"
                plt.savefig(reach_plot_path)
                plt.close()
                print(f"Saved boxplot for reachability to {reach_plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate or save boxplot for reachability: {e}")
        else:
            print("Info: 'reachability' column not found in DataFrame. Skipping reachability boxplot.")

        # Generate and save ENHANCED summary statistics
        try:
            # Select columns for which to generate describe() stats
            cols_to_describe = ['final', 'auc', 'step80', 'reachability']
            # Filter out columns not actually present in the DataFrame to avoid errors
            existing_cols_to_describe = [col for col in cols_to_describe if col in df.columns]
            
            if existing_cols_to_describe:
                summary_stats = df.groupby("topology")[existing_cols_to_describe].describe()
                # Transpose for better readability if there are many stats per topology
                # summary_stats = summary_stats.transpose()
                stats_path = plots_dir / "detailed_summary_stats.csv"
                summary_stats.to_csv(stats_path)
                print(f"Saved detailed summary stats to {stats_path}")
                
                # Also print reachability describe to console for quick view
                if 'reachability' in existing_cols_to_describe:
                    print("\nReachability Statistics by Topology:")
                    print(summary_stats['reachability'])
            else:
                print("Info: No columns selected for description were found in DataFrame. Skipping detailed summary stats.")

        except Exception as e:
            print(f"Warning: Could not generate or save detailed summary stats: {e}")

if __name__ == "__main__":
    main() 