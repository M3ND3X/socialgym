import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentManager:
    def __init__(self, db_path: str = "data/raw/experiment_data.db"):
        """Initialize experiment manager with database connection"""
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for experiment tracking"""
        logging.basicConfig(
            filename='data/logs/experiment_logs.csv',
            level=logging.INFO,
            format='%(asctime)s,%(levelname)s,%(message)s'
        )

    def setup_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    framing TEXT,
                    social_context TEXT,
                    payoff_structure TEXT,
                    agent_instruction TEXT,
                    agent_name TEXT,
                    persona TEXT,
                    bias TEXT,
                    temperature REAL,
                    max_output_tokens INTEGER,
                    top_p REAL,
                    model TEXT,
                    top_k INTEGER,
                    stop_sequences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create iterations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    iteration_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

            # Create rounds table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rounds (
                    round_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_id INTEGER,
                    round_number INTEGER,
                    system_prompt TEXT,
                    user_prompt TEXT,
                    raw_model_response TEXT,
                    cost REAL,
                    generated_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (iteration_id) REFERENCES iterations (iteration_id)
                )
            ''')

    def create_experiment(self, config: Dict[str, Any]) -> int:
        """Create a new experiment and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Extract fields from config
            fields = [
                'framing', 'social_context', 'payoff_structure',
                'agent_instruction', 'agent_name', 'persona', 'bias',
                'temperature', 'max_output_tokens', 'top_p', 'model',
                'top_k', 'stop_sequences'
            ]
            
            values = [config.get(field) for field in fields]
            
            # Convert lists/dicts to JSON strings
            values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in values]
            
            query = f'''
                INSERT INTO experiments 
                ({','.join(fields)})
                VALUES ({','.join(['?' for _ in fields])})
            '''
            
            cursor.execute(query, values)
            experiment_id = cursor.lastrowid
            
            logging.info(f"Created experiment {experiment_id} with config: {config}")
            return experiment_id

    def create_iteration(self, experiment_id: int, iteration_number: int) -> int:
        """Create a new iteration for an experiment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO iterations (experiment_id, iteration_number)
                VALUES (?, ?)
            ''', (experiment_id, iteration_number))
            return cursor.lastrowid

    def record_round(self, iteration_id: int, round_data: Dict[str, Any]) -> int:
        """Record a round's results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            fields = [
                'round_number', 'system_prompt', 'user_prompt',
                'raw_model_response', 'cost', 'generated_tokens'
            ]
            
            values = [round_data.get(field) for field in fields]
            values.insert(0, iteration_id)  # Add iteration_id
            
            query = f'''
                INSERT INTO rounds 
                (iteration_id, {','.join(fields)})
                VALUES ({','.join(['?' for _ in range(len(fields) + 1)])})
            '''
            
            cursor.execute(query, values)
            return cursor.lastrowid

    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """Retrieve all results for an experiment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get experiment details
            cursor.execute('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            experiment = cursor.fetchone()
            
            # Get all iterations and rounds
            cursor.execute('''
                SELECT i.iteration_number, r.*
                FROM iterations i
                JOIN rounds r ON i.iteration_id = r.iteration_id
                WHERE i.experiment_id = ?
                ORDER BY i.iteration_number, r.round_number
            ''', (experiment_id,))
            rounds = cursor.fetchall()
            
            return {
                'experiment': experiment,
                'rounds': rounds
            } 