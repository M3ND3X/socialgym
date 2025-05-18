#!/usr/bin/env python
import os
import json
import time
from pathlib import Path
import argparse
import logging
from datetime import datetime
import sys
from typing import List, Dict, Any, Tuple
import queue
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar
import concurrent.futures  # Import for parallel execution

# Import from the thesis module
from experiment_runner import ExperimentRunner
from main import run_experiment, save_to_database

def setup_logging(log_file: str) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_api_key() -> str:
    """Load the paid API key from .env file"""
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY1')
    
    if not api_key:
        raise ValueError("No API key found in .env file under GOOGLE_API_KEY1")
    
    print("Loaded paid API key")
    return api_key

def handle_api_limit(output_queue=None):
    """Handle API rate limit by sleeping"""
    message = "API rate limit reached. Sleeping for 30 seconds before continuing..."
    print(message)
    if output_queue:
        output_queue.put(message)
    time.sleep(30)  # Sleep for 30 seconds
    message = "Resuming after rate limit pause"
    print(message)
    if output_queue:
        output_queue.put(message)

def load_configs() -> Dict[str, Any]:
    """Load experiment configurations from config/ directory"""
    config_dir = Path("config")
    configs = {}
    
    # Load all JSON configs
    for config_file in config_dir.glob("*.json"):
        if config_file.name in ("model_config.json", "memory_iterative.json"):  # Skip model and memory_iterative configs
            continue
        with open(config_file, 'r') as f:
            configs[config_file.stem] = json.load(f)
    
    return configs

def get_conditions(config: Dict[str, Any]) -> List[str]:
    """Extract conditions from a configuration"""
    conditions = []
    if "conditions" in config:
        if isinstance(config["conditions"], dict):
            conditions = list(config["conditions"].keys())
        elif isinstance(config["conditions"], list):
            conditions = config["conditions"]
    
    if not conditions:
        conditions = ["neutral"]  # Default condition
    
    return conditions

def get_memory_types(exp_type: str) -> List[str]:
    """Get memory types based on experiment type"""
    if exp_type == "memory_iterative":
        # For memory_iterative, don't add memory types
        return ["no_memory"]  # Just use one type as it's handled internally
    else:
        # For other experiment types, use all memory types
        return ["no_memory", "short_term_memory", "long_term_memory"]

def get_models() -> List[str]:
    """Get the list of models to run experiments with"""
    with open("config/model_config.json", 'r') as f:
        model_config = json.load(f)
    
    return list(model_config["models"].keys())

def save_checkpoint(checkpoint_file: str, exp_type: str, condition: str, memory_type: str) -> None:
    """Save the current experiment state to a checkpoint file"""
    checkpoint = {
        "exp_type": exp_type,
        "condition": condition,
        "memory_type": memory_type,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load the last experiment checkpoint"""
    if not os.path.exists(checkpoint_file):
        return {}
    
    with open(checkpoint_file, 'r') as f:
        return json.load(f)

def get_experiment_list() -> List[Tuple[str, str, str]]:
    """Get a list of all experiment configurations"""
    configs = load_configs()
    experiment_list = []
    
    for exp_type, config in configs.items():
        conditions = get_conditions(config)
        memory_types = get_memory_types(exp_type)
        
        for condition in conditions:
            for memory_type in memory_types:
                experiment_list.append((exp_type, condition, memory_type))
    
    return experiment_list

def run_single_experiment(experiment: Tuple[str, str, str], models: List[str], api_key: str, checkpoint_file: str, 
                          db_path: str, configs: Dict[str, Any], output_queue: queue.Queue = None):
    """Run a single experiment configuration with all models"""
    exp_type, condition, memory_type = experiment
    config = configs[exp_type]
    
    print(f"\n{'-'*80}")
    print(f"Starting experiment:")
    print(f"Type: {exp_type}, Condition: {condition}, Memory: {memory_type}")
    print(f"Running with {len(models)} models")
    print(f"{'-'*80}\n")
    
    encountered_quota_error = False
    
    try:
        # Save checkpoint before running this experiment
        save_checkpoint(checkpoint_file, exp_type, condition, memory_type)
        
        # Create a copy of the config to avoid modifying the original
        exp_config = config.copy()
        
        # Add condition to experiment type for output directory
        experiment_with_condition = f"{exp_config['experiment_type']}_{condition}"
        exp_config['experiment_type'] = experiment_with_condition
        
        # Run experiment
        results = run_experiment(
            config=exp_config,
            models=models,
            num_rounds=50,  # 50 rounds for each experiment
            memory_type=memory_type,
            condition=condition,
            output_queue=output_queue,
        )
        
        # Save experiment results to database
        save_to_database(db_path, results, exp_config)
        
        # Process output from the queue if any
        if output_queue:
            while not output_queue.empty():
                msg = output_queue.get_nowait()
                print(msg)
                
                # Check for API quota exceeded message
                if "API quota exceeded" in msg or "rate limit" in msg.lower():
                    encountered_quota_error = True
                    handle_api_limit(output_queue)
        
        return True, encountered_quota_error
        
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Check if it was an API quota error
        if "API quota exceeded" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower():
            encountered_quota_error = True
            message = "API quota error detected in exception"
            print(message)
            if output_queue:
                output_queue.put(message)
            handle_api_limit(output_queue)
        
        return False, encountered_quota_error

def run_batch_experiments(continue_from_checkpoint: bool = False, parallel_count: int = 1, missing: bool = False):
    """Run all experiments in batch mode"""
    # Setup
    db_path = os.getenv('DB_PATH', 'data/raw/experiment_data.db')
    log_path = os.getenv('LOG_PATH', 'data/logs/batch_experiments.log')
    checkpoint_file = "data/checkpoint.json"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    os.makedirs("results/experiments", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/batch_experiment_{timestamp}.log"
    # Create logs directory before creating file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    setup_logging(log_file)
    
    # Load configurations, models, and API key
    configs = load_configs()
    all_models = get_models()
    api_key = load_api_key()
    
    # Set the API key in the environment
    os.environ['GOOGLE_API_KEY'] = api_key
    
    # Configure output queue for logging
    output_queue = queue.Queue()
    
    # Get the complete list of experiments
    all_experiments = get_experiment_list()
    total_experiments = len(all_experiments)
    # Initialize start index for sequential experiments
    start_idx = 0

    # Handle missing experiments: run only missing model runs and merge results
    if missing:
        import shutil
        base_dir = Path("results/experiments")
        # Build entries of missing tasks: (exp_type, condition, memory_type, missing_models, exp_dir)
        entries = []
        for exp_type, condition, memory_type in all_experiments:
            # Construct experiment identifier (suffix)
            exp_name = f"{configs[exp_type]['experiment_type']}_{condition}"
            # Find candidate directories matching this experiment+condition
            candidate_dirs = []
            if base_dir.exists():
                for d in base_dir.iterdir():
                    if not d.is_dir() or not d.name.endswith(f"_{exp_name}"):
                        continue
                    cfg_file = d / "config.json"
                    if not cfg_file.exists():
                        continue
                    try:
                        existing_cfg = json.load(open(cfg_file))
                    except:
                        continue
                    if (existing_cfg.get('experiment_type') == exp_name
                            and existing_cfg.get('memory_type') == memory_type
                            and existing_cfg.get('condition') == condition):
                        candidate_dirs.append(d)
            # Remove empty dirs that only have config.json
            cleaned_dirs = []
            for d in candidate_dirs:
                files = [f for f in d.iterdir() if f.is_file() and f.name != 'config.json']
                if not files:
                    shutil.rmtree(d)
                else:
                    cleaned_dirs.append(d)
            candidate_dirs = cleaned_dirs
            # Determine existing models across all candidate dirs
            existing_models = set()
            for d in candidate_dirs:
                for m in all_models:
                    if (d / f"{m}_{memory_type}_results.json").exists():
                        existing_models.add(m)
            missing_models = [m for m in all_models if m not in existing_models]
            if missing_models:
                # Choose an output directory if one exists
                exp_dir = candidate_dirs[0] if candidate_dirs else None
                entries.append((exp_type, condition, memory_type, missing_models, exp_dir))
        # Summarize before running
        print(f"Found {len(entries)} missing experiment configurations out of {total_experiments} total experiments.")
        if not entries:
            print("No missing models to run. Exiting.")
            return
        # Run missing entries, tracking progress
        total_runs = sum(len(item[3]) for item in entries)
        exp_pbar = tqdm(total=total_runs, desc="Missing Models Progress", position=0)
        from concurrent.futures import ThreadPoolExecutor
        # Worker task now takes a worker_id for logging and entries list
        def worker_task(worker_id, worker_entries):
            for exp_type, condition, memory_type, models_list, exp_dir in worker_entries:
                tqdm.write(f"[Worker {worker_id+1}] Running {len(models_list)} missing models for {exp_type}_{condition}_{memory_type}")
                exp_config = configs[exp_type].copy()
                exp_config['experiment_type'] = f"{exp_config['experiment_type']}_{condition}"
                results = run_experiment(
                    config=exp_config,
                    models=models_list,
                    num_rounds=50,
                    memory_type=memory_type,
                    condition=condition,
                    output_queue=output_queue,
                    results_dir=str(exp_dir) if exp_dir else None,
                )
                # Drain and print any queued output messages
                while not output_queue.empty():
                    msg = output_queue.get_nowait()
                    tqdm.write(f"[Worker {worker_id+1}] {msg}")
                results_dir_path = Path(results['results_dir'])
                combined_file = results_dir_path / "combined_results.json"
                merged = {}
                if combined_file.exists():
                    try:
                        merged = json.load(open(combined_file))
                    except:
                        merged = {}
                # Merge in new model outputs
                for m in models_list:
                    if m in results:
                        merged[m] = results[m]
                # Update metadata
                merged.update({
                    'timestamp': results.get('timestamp', int(time.time())),
                    'experiment_id': results.get('experiment_id'),
                    'config_file': results.get('config_file'),
                    'is_test': results.get('is_test'),
                    'results_dir': results.get('results_dir'),
                })
                with open(combined_file, 'w') as f:
                    json.dump(merged, f, indent=2)
                save_to_database(db_path, merged, exp_config)
                exp_pbar.update(len(models_list))
        # Execute in parallel or serial
        if parallel_count > 1:
            workers = [[] for _ in range(parallel_count)]
            for idx, entry in enumerate(entries):
                workers[idx % parallel_count].append(entry)
            with ThreadPoolExecutor(max_workers=parallel_count) as executor:
                for worker_id, w in enumerate(workers):
                    if w:
                        executor.submit(worker_task, worker_id, w)
        else:
            # Serial execution uses worker 1
            worker_task(0, entries)
        exp_pbar.close()
        return

    # Otherwise, not missing: optionally continue from last checkpoint
    if continue_from_checkpoint:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            last_exp_type = checkpoint.get('exp_type')
            last_condition = checkpoint.get('condition')
            last_memory_type = checkpoint.get('memory_type')
            print(f"Found checkpoint: {last_exp_type}, {last_condition}, {last_memory_type}")
            for i, (exp_type, condition, memory_type) in enumerate(all_experiments):
                if exp_type == last_exp_type and condition == last_condition and memory_type == last_memory_type:
                    start_idx = i + 1
                    break
            if start_idx >= len(all_experiments):
                print("All experiments already completed.")
                return
            print(f"Continuing from experiment {start_idx+1}/{total_experiments}")
    experiments_to_run = all_experiments[start_idx:]
    
    print(f"Preparing to run {len(experiments_to_run)} experiment configurations with {len(all_models)} models each")
    print(f"Each experiment will run for 50 rounds")
    print(f"Using a single paid API key")
    print(f"Logging to: {log_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    
    if parallel_count > 1:
        print(f"Running experiments in parallel with {parallel_count} workers")
        
        # Group models by family to distribute them optimally across workers
        model_groups = {}
        for model in all_models:
            # Extract model family (e.g., 'gemini', 'gemma', 'claude')
            prefix = model.split('-')[0].lower()
            if prefix not in model_groups:
                model_groups[prefix] = []
            model_groups[prefix].append(model)
        
        # Distribute models across workers to minimize rate limit issues
        worker_model_lists = [[] for _ in range(parallel_count)]
        
        # Assign models to workers in round-robin fashion grouped by family
        worker_idx = 0
        for family, models in model_groups.items():
            print(f"Distributing {len(models)} {family} models across {parallel_count} workers")
            for model in models:
                worker_model_lists[worker_idx].append(model)
                worker_idx = (worker_idx + 1) % parallel_count
        
        # Print distribution of models to workers
        for i, models in enumerate(worker_model_lists):
            print(f"Worker {i+1} will process {len(models)} models: {', '.join(models)}")
            
        # Create a progress bar for all experiments
        total_work = len(experiments_to_run) * len(all_models)
        exp_pbar = tqdm(total=total_work, desc="Total Progress", position=0)
        
        # Use a results dictionary to store results from each worker
        all_results = {}
        
        # Run experiments in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            futures = []
            
            # Submit work to workers based on model assignments
            for worker_id, worker_models in enumerate(worker_model_lists):
                if not worker_models:  # Skip empty model lists
                    continue
                    
                # Create a dedicated worker function to handle a subset of models
                def worker_task(worker_id, worker_models):
                    results = {}
                    encountered_quota_error = False
                    
                    for exp_idx, experiment in enumerate(experiments_to_run):
                        exp_type, condition, memory_type = experiment
                        config = configs[exp_type]
                        
                        print(f"\n[Worker {worker_id+1}] Starting experiment {exp_idx+1}/{len(experiments_to_run)}:")
                        print(f"Type: {exp_type}, Condition: {condition}, Memory: {memory_type}")
                        print(f"Models: {worker_models}")
                        
                        # If we encountered a quota error, sleep before continuing
                        if encountered_quota_error:
                            message = f"[Worker {worker_id+1}] Adding pause due to previous quota error..."
                            print(message)
                            if output_queue:
                                output_queue.put(message)
                            time.sleep(60)
                            encountered_quota_error = False
                        
                        try:
                            # Create a copy of the config
                            exp_config = config.copy()
                            experiment_with_condition = f"{exp_config['experiment_type']}_{condition}"
                            exp_config['experiment_type'] = experiment_with_condition
                            
                            # Run the experiment with this worker's set of models
                            worker_results = run_experiment(
                                config=exp_config,
                                models=worker_models,
                                num_rounds=50,
                                memory_type=memory_type,
                                condition=condition,
                                output_queue=output_queue,
                            )
                            
                            # Merge results
                            for model, model_result in worker_results.items():
                                exp_key = f"{exp_type}_{condition}_{memory_type}"
                                if exp_key not in results:
                                    results[exp_key] = {}
                                results[exp_key][model] = model_result
                            
                            # Update progress bar
                            exp_pbar.update(len(worker_models))
                            
                            # Check for API quota errors in the output queue
                            while not output_queue.empty():
                                msg = output_queue.get_nowait()
                                if "API quota exceeded" in msg or "rate limit" in msg.lower():
                                    encountered_quota_error = True
                                    #handle_api_limit(output_queue)
                            
                        except Exception as e:
                            print(f"[Worker {worker_id+1}] Error: {str(e)}")
                            traceback.print_exc()
                            
                            # Check if it was an API quota error
                            if "API quota exceeded" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower():
                                encountered_quota_error = True
                                message = f"[Worker {worker_id+1}] API quota error detected"
                                print(message)
                                if output_queue:
                                    output_queue.put(message)
                                    handle_api_limit(output_queue)
                            
                            # Still update progress even on error
                            exp_pbar.update(len(worker_models))
                    
                    return results, encountered_quota_error
                
                # Submit the worker task to the executor
                futures.append(executor.submit(worker_task, worker_id, worker_models))
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                worker_results, worker_quota_error = future.result()
                
                # Merge worker results into the main results
                for exp_key, exp_results in worker_results.items():
                    if exp_key not in all_results:
                        all_results[exp_key] = {}
                    all_results[exp_key].update(exp_results)
        
        # Close progress bar
        exp_pbar.close()
        
        # Save all results to database
        print("Saving all results to database...")
        for exp_key, exp_results in all_results.items():
            exp_type, condition, memory_type = exp_key.split('_')
            if exp_type in configs:
                exp_config = configs[exp_type].copy()
                exp_config['experiment_type'] = f"{exp_config['experiment_type']}_{condition}"
                save_to_database(db_path, exp_results, exp_config)
                
    else:
        # Run experiments sequentially (original behavior)
        encountered_quota_error = False  # Track if we encountered an API quota error

        # Create a progress bar for all experiments
        exp_pbar = tqdm(total=len(experiments_to_run), desc="Experiments", position=0)

        for i, (exp_type, condition, memory_type) in enumerate(experiments_to_run):
            config = configs[exp_type]
            exp_num = start_idx + i + 1

            print(f"\n{'='*80}\nStarting experiments for {exp_type}\n{'='*80}")
            print(f"\n{'-'*80}")
            print(f"Experiment {exp_num}/{total_experiments}:")
            print(f"Type: {exp_type}, Condition: {condition}, Memory: {memory_type}")
            print(f"Running with {len(all_models)} models")
            print(f"{'-'*80}\n")

            # If we encountered quota error in the previous experiment, add a 15-second sleep
            if encountered_quota_error:
                message = "Adding 15-second pause between experiments due to previous API quota error..."
                tqdm.write(message)
                if output_queue:
                    output_queue.put(message)
                time.sleep(30)
                encountered_quota_error = False  # Reset the flag

            # Create a progress bar for models within this experiment
            model_pbar = tqdm(total=len(all_models), desc=f"Models ({condition})", position=1, leave=False)

            try:
                # Save checkpoint before running this experiment
                save_checkpoint(checkpoint_file, exp_type, condition, memory_type)

                # Create a copy of the config to avoid modifying the original
                exp_config = config.copy()
                # Add condition to experiment type for output directory
                exp_config['experiment_type'] = f"{exp_config['experiment_type']}_{condition}"

                # Run experiment
                results = run_experiment(
                    config=exp_config,
                    models=all_models,
                    num_rounds=50,
                    memory_type=memory_type,
                    condition=condition,
                    output_queue=output_queue,
                )
                # Save experiment results to database
                save_to_database(db_path, results, exp_config)

                # Process and display output from the queue
                while not output_queue.empty():
                    msg = output_queue.get_nowait()
                    tqdm.write(msg)
                    if "API quota exceeded" in msg or "rate limit" in msg.lower():
                        encountered_quota_error = True
                        #handle_api_limit(output_queue)
            except Exception as e:
                tqdm.write(f"Error running experiment: {str(e)}")
                import traceback
                tqdm.write(traceback.format_exc())
                if "API quota exceeded" in str(e).lower() or "quota" in str(e).lower():
                    encountered_quota_error = True
                    message = "API quota error detected in exception"
                    tqdm.write(message)
                    if output_queue:
                        output_queue.put(message)
                        #handle_api_limit(output_queue)

            # Update the progress bars
            model_pbar.close()
            exp_pbar.update(1)

        exp_pbar.close()
        print(f"\n{'='*80}")
        print(f"All experiments completed.")
        print(f"Ran {len(experiments_to_run)} experiment configurations with {len(all_models)} models each")
        print(f"Using single paid API key")
        print(f"See logs for details: {log_file}")
        print(f"{'='*80}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run batch experiments for all configurations")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment configurations without running them")
    parser.add_argument("--c", "--continue", dest="continue_from_checkpoint", action="store_true", 
                      help="Continue from the last checkpoint")
    parser.add_argument("--p", "--parallel", dest="parallel_count", type=int, default=1,
                      help="Number of parallel experiments to run (default: 1)")
    parser.add_argument("-m", "--missing", dest="missing", action="store_true", help="Run only missing experiments (skip completed and memory_iterative)")
    parser.add_argument("--test", action="store_true", help="Run a simple test to verify setup")
    args = parser.parse_args()
    
    if args.test:
        # Simple test: run one round for the first model and config
        configs = load_configs()
        models = get_models()
        if not configs or not models:
            print("No configs or models found for test mode.")
            sys.exit(1)
        # Pick first config, condition, memory_type
        test_exp = list(configs.keys())[0]
        condition = get_conditions(configs[test_exp])[0]
        memory_type = get_memory_types(test_exp)[0]
        print(f"Running test for experiment {test_exp}, condition {condition}, memory {memory_type}, model {models[0]}")
        results = run_experiment(
            config={**configs[test_exp], "experiment_type": test_exp},
            models=[models[0]],
            num_rounds=1,
            memory_type=memory_type,
            condition=condition,
            is_test=True
        )
        print("Test results:", results)
        sys.exit(0)
    elif args.dry_run:
        # Just print the configurations that would be run
        configs = load_configs()
        models = get_models()
        
        print("Dry run mode - would run the following experiments:")
        for exp_type, config in configs.items():
            conditions = get_conditions(config)
            memory_types = get_memory_types(exp_type)
            
            for condition in conditions:
                for memory_type in memory_types:
                    print(f"Experiment: {exp_type}, Condition: {condition}, Memory: {memory_type}, Models: {len(models)}, Rounds: 50")
    else:
        # Actually run the experiments
        run_batch_experiments(continue_from_checkpoint=args.continue_from_checkpoint, 
                              parallel_count=args.parallel_count,
                              missing=args.missing)
