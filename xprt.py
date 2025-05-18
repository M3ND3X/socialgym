#!/usr/bin/env python3
import argparse
import json
import shutil
import time
import sys
from pathlib import Path

# Use script directory to build default path
script_dir = Path(__file__).parent

def merge_experiments(results_root: Path):
    """
    Merge multiple timestamped experiment folders into one per experiment+memory type.
    Moves model result files, deletes duplicate folders, and regenerates combined_results.json.
    """
    # Group directories by experiment_type and memory_type from their config.json
    groups = {}  # key: (exp_type_with_condition, memory_type) -> set of Path
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        cfg_file = d / 'config.json'
        if not cfg_file.exists():
            continue
        try:
            cfg = json.load(open(cfg_file))
        except Exception:
            print(f"Warning: could not load config.json in {d}, skipping")
            continue
        exp_name = cfg.get('experiment_type')  # includes condition suffix
        memory_type = cfg.get('memory_type')
        if not exp_name or not memory_type:
            continue
        key = (exp_name, memory_type)
        groups.setdefault(key, set()).add(d)
    # Process each group of (experiment_name, memory_type)
    for (exp_name, memory_type), dirs in groups.items():
        # Convert set of dirs to sorted list by folder name (timestamp prefix ensures order)
        dirs_sorted = sorted(dirs, key=lambda p: p.name)
        target_dir_old = dirs_sorted[0]
        # Ensure target_dir exists
        target_dir_old.mkdir(parents=True, exist_ok=True)
        # Merge duplicates if any
        for dup in dirs_sorted[1:]:
            # Move model result files into target_dir
            for f in dup.glob(f'*_{memory_type}_results.json'):
                dest = target_dir_old / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
            # Remove duplicate folder, skip if gone
            try:
                shutil.rmtree(dup)
            except FileNotFoundError:
                print(f"Warning: could not remove {dup}; directory not found, skipping.")
        # Rename directory to drop timestamp and include memory_type
        new_dir_name = f"{exp_name}_{memory_type}"
        new_dir = results_root / new_dir_name
        if target_dir_old.name != new_dir_name:
            # Remove any stale new_dir
            if new_dir.exists():
                shutil.rmtree(new_dir)
            target_dir_old.rename(new_dir)
            target_dir = new_dir
        else:
            target_dir = target_dir_old
        # Regenerate combined_results.json in target_dir
        combined = {}
        # Load each model result file
        for f in target_dir.glob(f'*_{memory_type}_results.json'):
            if f.name == 'combined_results.json':
                continue
            parts = f.stem.rsplit('_', 2)
            if len(parts) != 3 or parts[2] != 'results':
                continue
            model = parts[0]
            try:
                combined[model] = json.load(open(f))
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")
        # Add metadata
        combined.update({
            'timestamp': int(time.time()),
            'experiment_id': target_dir.name,
            'config_file': str(target_dir / 'config.json'),
            'results_dir': str(target_dir)
        })
        # Write combined_results.json
        combined_path = target_dir / 'combined_results.json'
        with open(combined_path, 'w') as cf:
            json.dump(combined, cf, indent=2)
        print(f"Merged into {target_dir} ({len(combined)} models, memory_type={memory_type})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge experiment result folders by experiment and memory type')
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=script_dir / 'results' / 'experiments',
        help='Root directory containing experiment run folders'
    )
    args = parser.parse_args()
    results_root = args.experiments_dir.resolve()
    if not results_root.exists():
        print(f"Error: experiments directory '{results_root}' does not exist.")
        sys.exit(1)
    print(f"Merging experiments from: {results_root}")
    merge_experiments(results_root) 