#!/usr/bin/env python
# aggregate_results.py
"""
Collect monitor.csv and training_summaries.yaml from a run folder
and write a tidy parquet that you can inspect or plot quickly.


python aggregate_results.py --results_dir results/topology_comparison_20250514_142447 --out pilot_cartpole.parquet

"""

import argparse, pathlib, yaml, pandas as pd, numpy as np

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
    df = walk_run_dir(pathlib.Path(args.results_dir))
    if df.empty:
        print("Warning: No data found. Output Parquet file will be empty or not created.")
    else:
        df.to_parquet(args.out)
        print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main() 