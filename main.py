import os
import json
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional
import logging
from experiment_runner import ExperimentRunner
import queue

# Rate limits per model (requests per minute)
MODEL_RPM_LIMITS = {
    'gemma': 30,
    'gemini-1.5-flash-8b': 2000,
    'gemini-1.5-flash': 2000,
    'gemini-1.5-pro': 1000,
    'gemini-2.0-flash-lite': 30000,
    'gemini-2.0-flash': 30000,
}

def get_model_rpm(model_name: str) -> int:
    for prefix, rpm in MODEL_RPM_LIMITS.items():
        if model_name.startswith(prefix):
            return rpm
    return None

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

def run_experiment(
    config: Dict[str, Any],
    models: List[str],
    num_rounds: int,
    memory_type: str = "no_memory",
    condition: str = "neutral",
    is_test: bool = False,
    output_queue: Optional[queue.Queue] = None,
    results_dir: Optional[str] = None  # Add parameter to accept an existing directory
) -> Dict[str, Any]:
    """Run experiment with given configuration"""
    max_retries = 3
    retry_count = 0
    
    while True:  # Infinite retry loop
        try:
            # Add parameters to config
            config["num_rounds"] = num_rounds
            config["memory_type"] = memory_type
            config["condition"] = condition
            config["condition_type"] = condition
            config["is_test"] = is_test
            
            # Use provided results directory or create a new one
            if results_dir:
                output_dir = Path(results_dir)
            else:
                # Create timestamp for this run
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Determine output directory - include condition in directory name
                base_dir = "results/test_runs" if is_test else "results/experiments"
                # Make sure experiment_type includes the condition if not already added
                experiment_type = config['experiment_type']
                if condition != "neutral" and condition not in experiment_type:
                    experiment_type = f"{experiment_type}_{condition}"
                output_dir = Path(base_dir) / f"{timestamp}_{experiment_type}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save configuration
                config_file = output_dir / "config.json"
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
            results = {}
            for model in models:
                if output_queue:
                    output_queue.put(f"Running experiment for model: {model}")
                
                model_retry_count = 0
                model_success = False
                
                while not model_success:
                    try:
                        # Create and run experiment for each model
                        runner = ExperimentRunner(
                            model_name=model,
                            config=config,
                            num_rounds=num_rounds,
                            results_dir=str(output_dir),
                            condition_type=condition,
                            memory_type=memory_type
                        )
                        
                        model_results = runner.run_experiment()
                        results[model] = model_results
                        
                        # Save individual model results including memory type
                        model_file = output_dir / f"{model}_{memory_type}_results.json"
                        with open(model_file, 'w') as f:
                            json.dump(model_results, f, indent=2)
                        
                        # Proactive throttle to respect model RPM and API keys
                        # Count API keys in environment
                        num_keys = 0
                        i = 1
                        while os.getenv(f"GOOGLE_API_KEY{i}"):
                            num_keys += 1
                            i += 1
                        rpm = get_model_rpm(model)
                        if rpm and num_keys:
                            delay = 60.0 / (rpm * num_keys)
                            time.sleep(delay)
                        model_success = True  # Mark as successful
                        
                    except Exception as e:
                        model_retry_count += 1
                        if model_retry_count < max_retries:
                            error_msg = f"Error running experiment for model {model}: {str(e)}. Retrying ({model_retry_count}/{max_retries})..."
                            print(error_msg)
                            if output_queue:
                                output_queue.put(error_msg)
                            time.sleep(5)  # Short delay before retry
                        else:
                            error_msg = f"Max retries reached for model {model}. Waiting 30 seconds before trying again..."
                            print(error_msg)
                            if output_queue:
                                output_queue.put(error_msg)
                            time.sleep(30)  # Longer delay after max retries
                            model_retry_count = 0  # Reset retry counter
                
            # Add metadata
            results.update({
                'timestamp': int(time.time()),
                'experiment_id': output_dir.name,
                'config_file': str(output_dir / "config.json"),
                'is_test': is_test,
                'results_dir': str(output_dir)
            })
            
            # Save combined results file with all model data
            combined_results_file = output_dir / "combined_results.json"
            with open(combined_results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            return results
            
        except Exception as e:
            retry_count += 1
            error_msg = f"Error in experiment setup: {str(e)}"
            if retry_count < max_retries:
                error_msg += f" Retrying ({retry_count}/{max_retries})..."
                print(error_msg)
                if output_queue:
                    output_queue.put(error_msg)
                time.sleep(5)
            else:
                error_msg += f" Max retries reached. Waiting 30 seconds before trying again..."
                print(error_msg)
                if output_queue:
                    output_queue.put(error_msg)
                time.sleep(30)
                retry_count = 0  # Reset retry counter

def save_to_database(db_path: str, results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save experiment results to database"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the full path to the combined results file
    results_dir = results.get('results_dir')
    combined_results_file = str(Path(results_dir) / "combined_results.json") if results_dir else ""
    
    # Prepare experiment data
    experiment_data = {
        'experiment_id': results.get('experiment_id'),
        'timestamp': results.get('timestamp', int(time.time())),
        'experiment_type': config.get('experiment_type'),
        'models': ','.join(config.get('models', [])),
        'num_rounds': config.get('num_rounds'),
        'memory_type': config.get('memory_type'),
        'condition': config.get('condition', 'neutral'),
        'status': 'completed',
        'results_file': combined_results_file,  # Use the full path to combined results
        'config_file': results.get('config_file'),
        'is_test': config.get('is_test', False)
    }
    
    try:
        # Make sure the results_file and config_file exist
        if experiment_data['results_file'] and not Path(experiment_data['results_file']).exists():
            print(f"Warning: Results file does not exist: {experiment_data['results_file']}")
        
        if experiment_data['config_file'] and not Path(experiment_data['config_file']).exists():
            print(f"Warning: Config file does not exist: {experiment_data['config_file']}")
        
        # Insert the experiment data
        cursor.execute("""
            INSERT INTO experiments (
                experiment_id, timestamp, experiment_type, models,
                num_rounds, memory_type, condition, status, results_file, 
                config_file, is_test
            ) VALUES (
                :experiment_id, :timestamp, :experiment_type, :models,
                :num_rounds, :memory_type, :condition, :status, :results_file,
                :config_file, :is_test
            )
        """, experiment_data)
        
        conn.commit()
        print(f"Successfully saved experiment to database with ID: {experiment_data['experiment_id']}")
        print(f"Results file saved as: {experiment_data['results_file']}")
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM decision-making experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to test")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument("--memory", choices=["no_memory", "short_term_memory", "long_term_memory"],
                      default="no_memory", help="Memory type")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Setup logging
    log_file = f"logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    
    # Run experiment
    results = run_experiment(
        config=config,
        models=args.models,
        num_rounds=args.rounds,
        memory_type=args.memory,
        is_test=args.test
    )
    
    # Save results
    if not args.test:
        save_to_database("data/experiment_data.db", results, config) 