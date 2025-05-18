#!/usr/bin/env python3
"""
flatten_results.py

CLI tool to flatten a directory of experiment-result JSON files into a tidy CSV, ready for analysis in Stata/R.
"""
import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Use script directory to locate test fixtures
SCRIPT_DIR = Path(__file__).parent


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON from a file, returning None on parse errors.
    """
    try:
        return json.load(file_path.open('r', encoding='utf-8'))
    except json.JSONDecodeError as e:
        logging.warning(f"Malformed JSON in {file_path}: {e}")
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
    return None


def generate_exp_id(folder: Path, model_name: str) -> str:
    """
    Generate a unique experiment ID as SHA-1 hash of folder path + model name.
    """
    h = hashlib.sha1()
    h.update(str(folder.resolve()).encode('utf-8'))
    h.update(b'|')
    h.update(model_name.encode('utf-8'))
    return h.hexdigest()


def add_lags(df: pd.DataFrame, vars_to_lag: List[str], group_key: str = 'exp_id') -> pd.DataFrame:
    """
    Add lag-1 columns for specified vars within each group_key.
    """
    df = df.sort_values([group_key, 'round'])
    for var in vars_to_lag:
        lag_name = f"{var}_lag1"
        df[lag_name] = df.groupby(group_key)[var].shift(1)
    return df


def flatten_results(input_dir: Path, output_csv: Path) -> None:
    """
    Traverse input_dir for *_results.json files, extract rounds, and write a single CSV.
    """
    # List to accumulate flattened rows
    rows: List[Dict[str, Any]] = []

    input_dir = input_dir.resolve()
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return
    logging.info(f"Scanning experiment folders in {input_dir}")

    # Each subdirectory under input_dir is one experiment_type_condition_memory_type
    for exp_dir in sorted(input_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Load folder-level config.json for metadata
        cfg_file = exp_dir / 'config.json'
        if not cfg_file.exists():
            logging.warning(f"Missing config.json in {exp_dir}, skipping")
            continue
        cfg = load_json(cfg_file)
        if cfg is None:
            continue
        exp_type = cfg.get('experiment_type')
        condition = cfg.get('condition')
        memory_type = cfg.get('memory_type')
        # Determine expected rounds value
        expected_rounds = cfg.get('num_rounds') or cfg.get('rounds')

        # Find all per-model result JSONs in this folder
        for json_file in exp_dir.glob(f'*_{memory_type}_results.json'):
            if not json_file.is_file():
                continue
            # Derive model name from filename
            parts = json_file.stem.rsplit(f"_{memory_type}_results", 1)
            model = parts[0] if parts else json_file.stem

            data = load_json(json_file)
            if data is None:
                continue
                
            # Handle the updated JSON structure - check for top-level "rounds" key directly
            recs = None
            if isinstance(data, dict):
                if 'rounds' in data and isinstance(data['rounds'], list):
                    recs = data['rounds']
                elif 'results' in data and isinstance(data['results'], list):
                    recs = data['results']
                # If still not found, this is likely the old structure, warn and skip
                else:
                    logging.warning(f"Unrecognized JSON structure in {json_file}, skipping")
                    continue
            elif isinstance(data, list):
                recs = data
            else:
                logging.warning(f"Unrecognized JSON structure in {json_file}, skipping")
                continue

            # Validate round counts
            if expected_rounds is not None and len(recs) != expected_rounds:
                logging.warning(
                    f"{exp_dir.name}/{json_file.name}: expected {expected_rounds} rounds, found {len(recs)}"
                )

            # Use model name from JSON if available, otherwise use filename
            if isinstance(data, dict) and 'model' in data:
                model = data['model']

            # Unique experiment-game ID per folder+model
            exp_id = generate_exp_id(exp_dir, model)

            # Flatten each round
            for rec in recs:
                rnum = rec.get('round') or rec.get('round_number')
                pc = rec.get('player_choice')
                ptc = rec.get('partner_choice')
                player_choice = 1 if pc == 'A' else (0 if pc == 'B' else pd.NA)
                partner_choice = 1 if ptc == 'A' else (0 if ptc == 'B' else pd.NA)

                row = {
                    'exp_id': exp_id,
                    'model': model,
                    'experiment_type': exp_type,
                    'condition': condition,
                    'memory_type': memory_type,
                    'round': rnum,
                    'player_choice': player_choice,
                    'partner_choice': partner_choice,
                    'player_score': rec.get('player_score', pd.NA),
                    'partner_score': rec.get('partner_score', pd.NA),
                    'total_tokens_used': rec.get('total_tokens_used', pd.NA),
                    'timestamp': rec.get('timestamp', pd.NA)
                }
                rows.append(row)

    # After iterating all folders and appending rows, build the DataFrame
    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("No rows to write. Exiting.")
        return

    # Add lag-1 variables for player and partner choices
    df = add_lags(df, ['player_choice', 'partner_choice'])

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write the flattened DataFrame to CSV
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Written {len(df)} rows to {output_csv}")


def run_smoke_test() -> None:
    """
    Smoke test using two fixture JSONs under tests/fixtures.
    Asserts expected row count.
    """
    fixture_dir = SCRIPT_DIR / 'tests' / 'fixtures'
    tmp_csv = SCRIPT_DIR / 'tests' / 'output.csv'
    flatten_results(fixture_dir, tmp_csv)
    df = pd.read_csv(tmp_csv)
    expected = 2 * 3  # e.g., 2 files x 3 rounds each in fixtures
    assert len(df) == expected, f"Smoke test failed: expected {expected} rows, got {len(df)}"
    logging.info("Smoke test passed!")
    tmp_csv.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Flatten experiment JSON results into a single CSV.'
    )
    parser.add_argument(
        '--input', '-i', type=Path, required=True,
        help='Root directory containing *_results.json files'
    )
    parser.add_argument(
        '--output', '-o', type=Path, required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--loglevel', default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run smoke test instead of flattening'
    )
    args = parser.parse_args()

    # Configure logging
    level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    if args.test:
        run_smoke_test()
        sys.exit(0)

    input_dir = args.input.resolve()
    output_csv = args.output.resolve()
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    flatten_results(input_dir, output_csv) 