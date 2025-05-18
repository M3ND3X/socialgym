import argparse
import json
from pathlib import Path
from experiment_runner import ExperimentRunner
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
from datetime import datetime
import os
import time
import random

def get_failed_models_from_experiment(experiment_dir: str) -> List[str]:
    """Get list of failed models from a previous experiment run"""
    failed_models = []
    
    # Load all result files from the experiment directory
    result_files = Path(experiment_dir).glob("*_results.json")
    
    # First try to load from summary results
    try:
        with open(Path(experiment_dir) / "summary.json", 'r') as f:
            summary = json.load(f)
            for model_name, result in summary.get("memory_iterative", {}).items():
                if result.get("status") == "failed":
                    failed_models.append(model_name)
            if failed_models:
                return failed_models
    except:
        pass
        
    # If no summary or no failed models found, check individual result files
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                
            # Extract model name from filename
            model_name = result_file.stem.replace("_results", "")
            
            # Check multiple possible failure indicators
            failed = False
            if isinstance(result, dict):
                # Check top-level status
                if result.get("status") == "failed":
                    failed = True
                # Check for error field
                elif "error" in result:
                    failed = True
                # Check if required fields are missing
                elif not result.get("rounds") or not result.get("summary"):
                    failed = True
                    
            if failed:
                failed_models.append(model_name)
                
        except Exception as e:
            # If we can't read the file or it's invalid JSON, consider it failed
            model_name = result_file.stem.replace("_results", "")
            failed_models.append(model_name)
            print(f"Error reading {result_file}: {e}")
            
    return failed_models

def find_latest_experiment(config_name: str) -> str:
    """Find the most recent experiment directory for given config"""
    results_dir = Path("results/test_runs")
    
    # Get all experiment directories for this config
    experiment_dirs = [d for d in results_dir.glob(f"*_{config_name}")]
    
    if not experiment_dirs:
        raise ValueError(f"No previous experiments found for {config_name}")
        
    # Sort by timestamp and get latest
    latest_dir = sorted(experiment_dirs, key=lambda x: x.name)[-1]
    return str(latest_dir)

def apply_config_options(config: Dict[str, Any], condition_type: str = None, memory_type: str = None) -> Dict[str, Any]:
    """Apply condition and memory options to config"""
    # Apply condition if specified and supported
    if "conditions" in config and condition_type:
        if condition_type in config["conditions"]:
            config["condition_type"] = condition_type
            print(f"Applied condition '{condition_type}' to experiment")
        else:
            print(f"Warning: Condition '{condition_type}' not found in config conditions: {list(config['conditions'].keys())}")
    
    # Apply memory type if specified
    if memory_type:
        valid_memory_types = ["no_memory", "short_term_memory", "long_term_memory"]
        if memory_type in valid_memory_types:
            # For memory_iterative config, use existing memory_type field
            if config.get("experiment_type") == "memory_iterative":
                config["memory_type"] = memory_type
            else:
                # For other configs, add memory settings
                config["memory_settings"] = {
                    "type": memory_type,
                    "enabled": True if memory_type != "no_memory" else False,
                    "max_rounds": 5 if memory_type == "short_term_memory" else -1,
                    "format": "You have played {n} rounds. In previous rounds:\n{history}"
                }
            print(f"Applied memory type '{memory_type}' to experiment")
        else:
            print(f"Warning: Invalid memory type '{memory_type}'. Valid options: {valid_memory_types}")
            
    return config

def run_tests(
    config_name: str,
    num_rounds: int = 1,
    num_configs: int = 1,
    continue_experiment: bool = False,
    condition_type: str = None,
    memory_type: str = None,
    models: List[str] = None
) -> Tuple[Dict[str, Any], str]:
    """Run tests with specified configuration"""
    
    # Set default models if none provided
    if models is None:
        models = ["gemini-1.0-pro"]
        
    # Load experiment config
    config_file = f"config/{config_name}.json"
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found: {config_file}")
        
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/test_runs/{timestamp}_{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}
    results[config_name] = {}
    
    # Run experiment for each model
    for model_name in models:
        try:
            # Initialize experiment runner
            runner = ExperimentRunner(
                model_name=model_name,
                config=config,
                num_rounds=num_rounds,
                results_dir=results_dir,
                condition_type=condition_type,
                memory_type=memory_type
            )
            
            # Run experiment
            experiment_results = runner.run_experiment()
            
            # Store results
            results[config_name][model_name] = {
                "status": "success",
                "model_family": model_name.split('-')[0],
                "experiment_id": experiment_results.get("experiment_id"),
                "summary": experiment_results.get("summary", {}),
                "results_file": experiment_results.get("results_file"),
                "config_file": config_file
            }
            
        except Exception as e:
            results[config_name][model_name] = {
                "status": "failed",
                "model_family": model_name.split('-')[0],
                "error": str(e)
            }
            print(f"Error testing {model_name}: {str(e)}")
            
    return results, results_dir

def display_results(results: Dict[str, Any], results_dir: str):
    """Display test results in a formatted way"""
    print("\nTest Results:")
    print("=" * 100)
    print(f"Results directory: {results_dir}")
    print("=" * 100)
    
    for config_name, config_results in results.items():
        print(f"\nConfig: {config_name}")
        print("=" * 50)
        
        # Group models by family
        family_groups = {}
        for model_name, result in config_results.items():
            family = result["model_family"]
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append((model_name, result))
        
        # Display results by family
        for family, models in family_groups.items():
            print(f"\n{family.upper()} Models:")
            print("-" * 40)
            
            for model_name, result in models:
                status = "+" if result["status"] == "success" else "x"
                print(f"\n{status} {model_name}")
                
                if result["status"] == "failed":
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  Experiment ID: {result['experiment_id']}")
                    if "summary" in result:
                        summary = result["summary"]
                        choices = summary.get("choices", {})
                        total_rounds = summary.get("total_rounds", 0)
                        print(f"  Total Rounds: {total_rounds}")
                        if choices:
                            print(f"  Choice Distribution:")
                            print(f"    Option A: {choices.get('A', 0)}")
                            print(f"    Option B: {choices.get('B', 0)}")
                            if total_rounds > 0:
                                coop_rate = choices.get('A', 0) / total_rounds
                                print(f"  Cooperation Rate: {coop_rate:.2%}")
                        token_usage = summary.get("token_usage", {})
                        if token_usage:
                            print(f"  Total Tokens: {token_usage.get('total_tokens', 0)}")
                    
                    if "results_file" in result:
                        print(f"  Results file: {result['results_file']}")
        
        print("\n" + "=" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utest", action="store_true", help="Run unit tests")
    parser.add_argument("--config", help="Configuration to use")
    parser.add_argument("--r", type=int, help="Number of rounds")
    parser.add_argument("--memory-type", choices=["no_memory", "short_term_memory", "long_term_memory"],
                       help="Memory type for experiments")
    parser.add_argument("--condition", help="Condition type to use")
    parser.add_argument("--models", help="Comma-separated list of models to test")
    
    args = parser.parse_args()
    
    if args.utest:
        config_name = args.config
        
        # Run the tests
        results, results_dir = run_tests(
            config_name=config_name,
            num_rounds=args.r,
            num_configs=1,
            continue_experiment=False,
            condition_type=args.condition,
            memory_type=args.memory_type,
            models=args.models.split(",") if args.models else ["gemini-1.0-pro"]
        )
        
        # Display results
        display_results(results, results_dir)
        
        return results, results_dir

if __name__ == "__main__":
    main() 