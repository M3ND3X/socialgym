import customtkinter as ctk
import json
from pathlib import Path
from typing import Dict, Any, List
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
from datetime import datetime
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from dotenv import load_dotenv
from main import run_experiment, save_to_database
import sqlite3
import time
import traceback
import sys

class LLMExperimentApp:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get database path from environment
        self.db_path = os.getenv('DB_PATH', 'data/raw/experiment_data.db')
        self.log_path = os.getenv('LOG_PATH', 'data/logs/experiment_logs.csv')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs("results/experiments", exist_ok=True)  # Ensure experiments directory exists
        
        # Initialize database if it doesn't exist
        self._init_database()
        
        # Initialize window
        self.window = ctk.CTk()
        self.window.title("SocialGYM")
        self.window.geometry("1400x800")
        self.window.minsize(1200, 600)
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize state
        self.experiment_running = False
        self.output_queue = queue.Queue()
        self.model_vars = {}
        self.api_keys = []
        self.is_test_mode = False
        self.experiment_thread = None  # Track the experiment thread
        self.current_results = {}  # Store current experiment results
        self.experiment_subprocess_pid = None  # Store PID of experiment subprocess
        self.history_loaded = False  # Track if history has been loaded
        
        # State management
        self.current_config = {
            "experiment_type": "memory_iterative",
            "memory_type": "no_memory",
            "num_rounds": 10,
            "models": [],
            "api_keys": []
        }
        
        # Load configs and saved state
        self.load_configs()
        self.load_saved_state()
        
        # Create layout
        self.create_layout()
        
        # Start output processing
        self.process_output()
        
        # Load experiment history when UI is ready
        self.window.after(2000, self.initial_load_history)
        
    def _init_database(self):
        """Initialize database with required tables"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    experiment_type TEXT NOT NULL,
                    models TEXT,
                    num_rounds INTEGER NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'no_memory',
                    condition TEXT NOT NULL DEFAULT 'neutral',
                    status TEXT NOT NULL DEFAULT 'completed',
                    results_file TEXT,
                    config_file TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_test BOOLEAN DEFAULT 0
                )
            """)
            
            # Check if the table already has data
            cursor.execute("SELECT COUNT(*) FROM experiments")
            count = cursor.fetchone()[0]
            if count == 0:
                print("Database is empty, no previous experiments found")
            else:
                print(f"Database contains {count} previous experiments")
            
            conn.commit()
            
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            traceback.print_exc()
            conn.rollback()
        finally:
            conn.close()

    def load_configs(self):
        """Load all configuration files"""
        config_dir = Path("config")
        self.configs = {}
        
        # Load all JSON configs
        for config_file in config_dir.glob("*.json"):
            with open(config_file, 'r') as f:
                self.configs[config_file.stem] = json.load(f)
                
        # Load model config
        with open(config_dir / "model_config.json", 'r') as f:
            self.model_config = json.load(f)

    def load_saved_state(self):
        """Load saved application state"""
        try:
            with open("app_state.json", "r") as f:
                saved_state = json.load(f)
                self.current_config.update(saved_state)
        except FileNotFoundError:
            pass

    def save_state(self):
        """Save current application state"""
        state = {
            "experiment_type": self.exp_type.get(),
            "memory_type": self.memory_var.get(),
            "num_rounds": self.rounds_var.get(),
            "models": [name for name, var in self.model_vars.items() if var.get()],
            "api_keys": [key.get() for key in self.api_keys if key.get()]
        }
        
        with open("app_state.json", "w") as f:
            json.dump(state, f)

    def create_layout(self):
        """Create main application layout"""
        # Configure main window grid
        self.window.grid_columnconfigure(0, weight=1)  # Left half
        self.window.grid_columnconfigure(1, weight=1)  # Right half
        self.window.grid_rowconfigure(0, weight=3)     # Top section
        self.window.grid_rowconfigure(1, weight=1)     # Bottom section (History)

        # Left half - Top section (Setup and Monitor)
        left_top = ctk.CTkFrame(self.window)
        left_top.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure left_top grid to make columns equal width
        left_top.grid_columnconfigure(0, weight=1)  # Setup section
        left_top.grid_columnconfigure(1, weight=1)  # Monitor section
        left_top.grid_rowconfigure(0, weight=1)     # Make row expand
        
        # Setup section
        self.setup_frame = ctk.CTkFrame(left_top)
        self.setup_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Monitor section
        self.monitor_frame = ctk.CTkFrame(left_top)
        self.monitor_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Left half - Bottom section (History)
        self.history_frame = ctk.CTkFrame(self.window)
        self.history_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure history frame to expand
        self.history_frame.grid_columnconfigure(0, weight=1)
        self.history_frame.grid_rowconfigure(0, weight=1)
        
        # Right half - Results
        self.results_frame = ctk.CTkFrame(self.window)
        self.results_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        # Configure results frame to expand
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)

        # Create content for each panel
        self.create_setup_panel()
        self.create_monitor_panel()
        self.create_history_panel()
        self.create_results_panel()

    def create_setup_panel(self):
        """Create the setup panel"""
        # Configure grid
        self.setup_frame.grid_columnconfigure(0, weight=1)
        
        # Experiment Type
        exp_label = ctk.CTkLabel(self.setup_frame, text="Experiment Type:")
        exp_label.pack(padx=5, pady=2)
        
        self.exp_type = ctk.CTkComboBox(
            self.setup_frame,
            values=list(self.configs.keys()),
            width=300  # Fixed initial width
        )
        self.exp_type.pack(fill="x", padx=5, pady=2)
        self.exp_type.set(self.current_config.get("experiment_type", "memory_iterative"))

        # Condition dropdown
        condition_label = ctk.CTkLabel(self.setup_frame, text="Condition:")
        condition_label.pack(padx=5, pady=2)
        
        self.condition_var = tk.StringVar(value="neutral")
        self.condition_dropdown = ctk.CTkComboBox(
            self.setup_frame,
            variable=self.condition_var,
            values=["neutral"],
            width=300
        )
        self.condition_dropdown.pack(fill="x", padx=5, pady=2)
        
        # Update conditions when experiment type changes
        self.exp_type.configure(command=self.on_experiment_type_change)

        # Memory Type
        memory_label = ctk.CTkLabel(self.setup_frame, text="Memory Type:")
        memory_label.pack(padx=5, pady=2)
        
        memory_frame = ctk.CTkFrame(self.setup_frame)
        memory_frame.pack(fill="x", padx=5, pady=2)
        
        self.memory_var = tk.StringVar(value=self.current_config.get("memory_type", "no_memory"))
        memory_types = ["no_memory", "short_term_memory", "long_term_memory"]
        
        for memory_type in memory_types:
            rb = ctk.CTkRadioButton(
                memory_frame,
                text=memory_type.replace("_", " ").title(),
                variable=self.memory_var,
                value=memory_type
            )
            rb.pack(side="left", padx=5)

        # Rounds Configuration
        rounds_frame = ctk.CTkFrame(self.setup_frame)
        rounds_frame.pack(fill="x", padx=5, pady=5)
        
        rounds_label = ctk.CTkLabel(rounds_frame, text="Number of Rounds:")
        rounds_label.pack(side="left", padx=5)
        
        self.rounds_var = tk.StringVar(value=self.current_config.get("num_rounds", "20"))
        rounds_entry = ctk.CTkEntry(
            rounds_frame,
            textvariable=self.rounds_var,
            width=60
        )
        rounds_entry.pack(side="left", padx=5)

        # Models section with scrollable frame
        models_frame = ctk.CTkScrollableFrame(
            self.setup_frame,
            label_text="Models:",
            height=200  # Match history panel height
        )
        models_frame.pack(fill="x", padx=5, pady=5)
        
        # Add model checkboxes
        for model_name in self.model_config["models"].keys():
            var = tk.BooleanVar(value=model_name in self.current_config.get("models", []))
            self.model_vars[model_name] = var
            
            checkbox = ctk.CTkCheckBox(
                models_frame,
                text=model_name,
                variable=var
            )
            checkbox.pack(anchor="w", padx=5, pady=2)

        # Control buttons
        self.create_control_buttons()

    def create_control_buttons(self):
        """Create control buttons"""
        # Create button frame that aligns with history panel
        button_frame = ctk.CTkFrame(self.setup_frame)
        button_frame.pack(fill="x", padx=5, pady=(0, 5))  # Reduced bottom padding
        
        # Add buttons
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="Start",
            command=self.start_experiment
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.history_btn = ctk.CTkButton(
            button_frame,
            text="History",
            command=self.create_history_viewer
        )
        self.history_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="Stop",
            command=self.stop_experiment,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

    def show_experiment_setup(self):
        """Show experiment setup panel"""
        self.create_experiment_setup()
        
    def show_model_selection(self):
        """Show model selection panel"""
        # Clear main content
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Create model selection frame
        model_frame = ctk.CTkFrame(self.main_content)
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Model Selection
        model_label = ctk.CTkLabel(
            model_frame, 
            text="Select Models:",
            font=("Arial", 16, "bold")
        )
        model_label.pack(padx=10, pady=5)
        
        # Create scrollable frame for models
        scroll_frame = ctk.CTkScrollableFrame(model_frame)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add checkboxes for each model
        self.model_vars = {}
        for model_name in self.model_config["models"].keys():
            var = tk.BooleanVar(value=False)
            self.model_vars[model_name] = var
            
            checkbox = ctk.CTkCheckBox(
                scroll_frame,
                text=model_name,
                variable=var,
                width=200
            )
            checkbox.pack(padx=10, pady=2, anchor="w")
            
        # API Key Management
        key_frame = ctk.CTkFrame(model_frame)
        key_frame.pack(fill="x", padx=10, pady=10)
        
        key_label = ctk.CTkLabel(
            key_frame,
            text="API Keys:",
            font=("Arial", 14, "bold")
        )
        key_label.pack(padx=10, pady=5)
        
        # API Key entries
        self.api_keys = []
        for i in range(5):
            key_entry = ctk.CTkEntry(
                key_frame,
                width=300,
                placeholder_text=f"API Key {i+1}",
                show="*"
            )
            key_entry.pack(padx=10, pady=2)
            self.api_keys.append(key_entry)
            
        # Save button
        save_btn = ctk.CTkButton(
            key_frame,
            text="Save API Keys",
            command=self.save_api_keys,
            width=150
        )
        save_btn.pack(pady=10)

    def show_live_monitor(self):
        """Show live monitor panel"""
        # Clear main content
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Create monitor frame
        monitor_frame = ctk.CTkFrame(self.main_content)
        monitor_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Progress section
        progress_frame = ctk.CTkFrame(monitor_frame)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Progress: Not Running",
            font=("Arial", 14)
        )
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        # Output log
        log_frame = ctk.CTkFrame(monitor_frame)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Output Log:",
            font=("Arial", 14, "bold")
        )
        log_label.pack(pady=5)
        
        self.output_text = ctk.CTkTextbox(
            log_frame,
            wrap="word",
            height=200
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Current stats table
        stats_frame = ctk.CTkFrame(monitor_frame)
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        stats_label = ctk.CTkLabel(
            stats_frame,
            text="Current Statistics:",
            font=("Arial", 14, "bold")
        )
        stats_label.pack(pady=5)
        
        # Create Treeview for stats
        self.stats_tree = ttk.Treeview(
            stats_frame,
            columns=("Model", "Round", "Choice", "Partner", "Score"),
            show="headings",
            height=6
        )
        
        # Configure columns
        for col in ("Model", "Round", "Choice", "Partner", "Score"):
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=80)
            
        self.stats_tree.pack(fill="x", padx=5, pady=5)

    def show_results_view(self):
        """Show results view panel"""
        # Clear main content
        for widget in self.main_content.winfo_children():
            widget.destroy()
            
        # Create notebook for tabs
        notebook = ttk.Notebook(self.main_content)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ctk.CTkFrame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        # Create summary table
        self.summary_tree = ttk.Treeview(
            summary_frame,
            columns=("Model", "Cooperation", "Rounds", "Tokens"),
            show="headings",
            height=6
        )
        
        for col in ("Model", "Cooperation", "Rounds", "Tokens"):
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=100)
        
        self.summary_tree.pack(fill="x", padx=5, pady=5)
        
        # Add charts
        chart_frame = ctk.CTkFrame(summary_frame)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Sample data - replace with real data
        models = ["Model1", "Model2"]
        coop_rates = [0.6, 0.4]
        
        ax1.bar(models, coop_rates)
        ax1.set_title("Cooperation Rates")
        ax1.set_ylabel("Rate")
        
        choices = ["A", "B"]
        counts = [30, 70]
        
        ax2.pie(counts, labels=choices, autopct='%1.1f%%')
        ax2.set_title("Choice Distribution")
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Detailed results tab
        details_frame = ctk.CTkFrame(notebook)
        notebook.add(details_frame, text="Detailed Results")
        
        details_tree = ttk.Treeview(
            details_frame,
            columns=("Round", "Model", "Choice", "Reasoning", "Score"),
            show="headings"
        )
        
        for col in ("Round", "Model", "Choice", "Reasoning", "Score"):
            details_tree.heading(col, text=col)
            details_tree.column(col, width=100)
            
        details_tree.pack(fill="both", expand=True, padx=10, pady=5)

    def save_api_keys(self):
        """Save API keys to .env file"""
        keys = [key.get() for key in self.api_keys if key.get()]
        
        if not keys:
            messagebox.showwarning("Warning", "No API keys entered")
            return
            
        try:
            with open(".env", "w") as f:
                for i, key in enumerate(keys, 1):
                    f.write(f"GOOGLE_API_KEY{i}={key}\n")
            messagebox.showinfo("Success", "API keys saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save API keys: {str(e)}")

    def start_experiment(self):
        """Start the experiment with current configuration"""
        try:
            # Clear previous results
            self.clear_results()
            
            # Rest of the start_experiment code...
            exp_type = self.exp_type.get()
            
            # Load config
            config_path = Path("config") / f"{exp_type}.json"
            if not config_path.exists():
                messagebox.showerror("Error", f"Config file not found: {config_path}")
                return
            
            with open(config_path) as f:
                config = json.load(f)
            
            # Get selected models
            selected_models = [
                model for model, var in self.model_vars.items()
                if var.get()
            ]
            
            if not selected_models:
                messagebox.showerror("Error", "Please select at least one model")
                return
            
            # Get number of rounds
            num_rounds = int(self.rounds_var.get())
            
            # Get memory type
            memory_type = self.memory_var.get()
            
            # Get condition
            condition = self.condition_var.get()
            
            # Update UI
            self.output_text.insert("end", f"Starting experiment with config: {exp_type}\n")
            self.output_text.insert("end", f"Condition: {condition}\n")
            self.output_text.see("end")
            
            # Set experiment running flag
            self.experiment_running = True
            
            # Update button states
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
            # Clear previous results
            self.summary_tree.delete(*self.summary_tree.get_children())
            
            # Run experiment in separate thread
            self.experiment_thread = threading.Thread(
                target=lambda: self.run_experiment_thread(
                    config=config,
                    models=selected_models,
                    num_rounds=num_rounds,
                    memory_type=memory_type,
                    condition=condition
                )
            )
            self.experiment_thread.daemon = True
            self.experiment_thread.start()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.output_text.insert("end", f"{error_msg}\n")
            self.output_text.see("end")
            messagebox.showerror("Error", error_msg)

    def run_experiment_thread(self, config, models, num_rounds, memory_type, condition):
        """Run experiment in separate thread"""
        try:
            self.window.after(0, lambda: self.progress_label.configure(text="Progress: Running"))
            self.window.after(0, lambda: self.progress_bar.set(0))
            
            # Create single experiment directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = Path("results") / "experiments" / f"{timestamp}_{config['experiment_type']}_{condition}"
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config first
            config_file = experiment_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            total_steps = len(models) * num_rounds
            completed_steps = 0
            all_results = {}
            
            # Run experiments for all models in a single call
            try:
                # Check if experiment has been stopped
                if not self.experiment_running:
                    self.output_queue.put("Experiment stopped by user")
                    return
                
                # Don't store the process ID to avoid potential app termination
                # Instead, just log the start of the experiment
                self.output_queue.put(f"Starting experiment with {len(models)} models and {num_rounds} rounds")
                
                # Set timeout to avoid infinite waiting
                max_wait_time = 120  # seconds
                start_time = time.time()
                
                # Create a thread-safe Event to signal experiment completion
                experiment_completed = threading.Event()
                
                # Define a thread-safe container for results
                results_container = {}
                
                # Function to run the experiment in a separate thread
                def run_exp():
                    try:
                        results = run_experiment(
                            config=config,
                            models=models,
                            num_rounds=num_rounds,
                            memory_type=memory_type,
                            condition=condition,
                            is_test=self.is_test_mode,
                            output_queue=self.output_queue,
                            results_dir=str(experiment_dir)
                        )
                        # Store results and signal completion
                        results_container.update(results)
                        experiment_completed.set()
                    except Exception as e:
                        self.output_queue.put(f"Error in experiment thread: {str(e)}")
                        # Signal completion even on error
                        experiment_completed.set()
                
                # Start experiment in another thread
                exp_thread = threading.Thread(target=run_exp)
                exp_thread.daemon = True
                exp_thread.start()
                
                # Wait for experiment to complete or for the experiment_running flag to be set to False
                while not experiment_completed.is_set() and self.experiment_running:
                    # Check if we've been waiting too long
                    if time.time() - start_time > max_wait_time:
                        self.output_queue.put(f"Experiment taking longer than {max_wait_time} seconds, but continuing...")
                        break
                    
                    # Give the UI thread a chance to process events
                    time.sleep(0.1)
                
                # If experiment_running is False, the user has stopped the experiment
                if not self.experiment_running:
                    self.output_queue.put("Experiment was stopped by user")
                    return
                    
                # If experiment completed, get results
                if experiment_completed.is_set() and results_container:
                    combined_results = results_container
                    
                    # Update progress
                    completed_steps = total_steps
                    self.window.after(0, lambda: self.progress_bar.set(1.0))
                    
                    # Extract individual model results
                    for model in models:
                        if model in combined_results:
                            all_results[model] = combined_results[model]
                else:
                    self.output_queue.put("Experiment did not complete successfully")
                    return
                        
            except Exception as e:
                self.output_queue.put(f"Error running experiment: {str(e)}")
                traceback.print_exc()
                return
            
            # Check if experiment has been stopped
            if not self.experiment_running:
                self.output_queue.put("Experiment stopped by user, not saving results")
                return
                
            # Save combined results
            if not self.is_test_mode and all_results:
                # Create metadata for database
                combined_results = {
                    "experiment_id": experiment_dir.name,
                    "timestamp": int(time.time()),
                    "config": config,
                    "results": all_results,
                    "results_dir": str(experiment_dir),
                    "config_file": str(config_file)
                }
                
                # Save combined results file
                combined_results_file = experiment_dir / "combined_results.json"
                with open(combined_results_file, 'w') as f:
                    json.dump(combined_results, f, indent=2)
                
                # Update database with the full combined results file path
                combined_results["results_file"] = str(combined_results_file)
                save_to_database(self.db_path, combined_results, config)
                
                self.output_queue.put(f"Results saved to database and {combined_results_file}")
            
            # Store current results
            self.current_results = all_results
            
            # Update UI safely using a single after call
            def update_ui():
                try:
                    if hasattr(self, 'summary_frame') and self.summary_frame.winfo_exists():
                        self.update_results_display(all_results)
                    self.update_history()
                    self.update_detailed_summary(all_results)
                except Exception as e:
                    self.output_queue.put(f"Error updating UI: {str(e)}")
            
            # Schedule UI update
            self.window.after(0, update_ui)
            
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.experiment_running = False
            def reset_ui():
                try:
                    if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
                        self.start_btn.configure(state="normal")
                    if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                        self.stop_btn.configure(state="disabled")
                    if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                        self.progress_label.configure(text="Progress: Not Running")
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.set(0)
                except Exception as e:
                    print(f"Error resetting UI: {str(e)}")
            
            self.window.after(0, reset_ui)

    def update_results_display(self, results):
        """Update results display with experiment data"""
        try:
            # Clear existing results
            self.clear_results()
            
            # Store current results
            self.current_results = results
            
            # Verify widgets exist before updating
            if not hasattr(self, 'summary_frame') or not self.summary_frame.winfo_exists():
                print("Summary frame is not available")
                return
            
            # Create charts
            self.create_result_charts(self.charts_frame, results)
            
            # Update detailed summary
            self.update_detailed_summary(results)
            
        except Exception as e:
            print(f"Error updating results display: {str(e)}")
            traceback.print_exc()
            # Use after to show error dialog to avoid threading issues
            self.window.after(0, lambda: messagebox.showerror("Error", f"Failed to update results display: {str(e)}"))

    def update_stats(self, model: str, round_num: int, choice: str, partner: str = "-", score: str = "-"):
        """Update statistics table"""
        try:
            if not hasattr(self, 'stats_tree') or not self.stats_tree.winfo_exists():
                return
            
            self.stats_tree.insert("", "end", values=(
                model,
                round_num,
                choice,
                partner,
                score
            ))
            
            # Keep only the last 6 rows
            children = self.stats_tree.get_children()
            if len(children) > 6:
                self.stats_tree.delete(children[0])
                
        except Exception as e:
            print(f"Error updating stats: {str(e)}")

    def load_results(self, experiment_type: str):
        """Load and display experiment results"""
        try:
            # Find latest results directory
            results_dir = Path("results/test_runs")
            latest_dir = max(results_dir.glob(f"*_{experiment_type}"), key=os.path.getctime)
            
            # Load results file
            results_files = list(latest_dir.glob("*_results.json"))
            if not results_files:
                raise FileNotFoundError("No results files found")
            
            results = []
            for file in results_files:
                with open(file, 'r') as f:
                    results.append(json.load(f))
                
            # Process results
            summary = {
                "total_rounds": 0,
                "choices": {"A": 0, "B": 0},
                "models": {}
            }
            
            for result in results:
                model_name = result.get("model", "unknown")
                rounds = result.get("summary", {}).get("total_rounds", 0)
                choices = result.get("summary", {}).get("choices", {})
                
                summary["models"][model_name] = {
                    "rounds": rounds,
                    "cooperation_rate": choices.get("A", 0) / (rounds if rounds > 0 else 1),
                    "tokens": result.get("summary", {}).get("token_usage", {}).get("total_tokens", 0)
                }
                
                summary["total_rounds"] += rounds
                summary["choices"]["A"] += choices.get("A", 0)
                summary["choices"]["B"] += choices.get("B", 0)
            
            # Update UI with processed results
            self.update_summary_table(summary)
            self.update_charts(summary)
            
        except Exception as e:
            self.output_queue.put(f"Error loading results: {str(e)}")

    def update_summary_table(self, results):
        """Update summary table with results"""
        # Clear existing items
        self.summary_tree.delete(*self.summary_tree.get_children())
        
        # Add new results
        for model, data in results.get("summary", {}).items():
            total_rounds = data.get("total_rounds", 0)
            choices = data.get("choices", {})
            if total_rounds > 0:
                coop_rate = choices.get("A", 0) / total_rounds
            else:
                coop_rate = 0
            
            self.summary_tree.insert("", "end", values=(
                model,
                f"{coop_rate:.2%}",
                total_rounds,
                data.get("token_usage", {}).get("total_tokens", 0)
            ))

    def update_charts(self, summary: Dict[str, Any]):
        """Update results charts"""
        try:
            # Get the chart frame
            chart_frame = self.results_notebook.winfo_children()[0].winfo_children()[1]
            
            # Clear existing charts
            for widget in chart_frame.winfo_children():
                widget.destroy()
            
            # Create new charts with actual data
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            plt.style.use('dark_background')
            
            # Cooperation rates
            models = list(summary.get("models", {}).keys())
            rates = [data["cooperation_rate"] for data in summary.get("models", {}).values()]
            
            if not models or not rates:
                # No data to display
                ctk.CTkLabel(chart_frame, text="No data available for visualization").pack(pady=20)
                return
            
            # Plot cooperation rates
            bars1 = ax1.bar(models, rates)
            ax1.set_title("Cooperation Rate by Model")
            ax1.set_ylabel("Cooperation Rate (%)")
            ax1.set_ylim(0, max(rates) + 0.15 if rates else 0.15)  # Increased padding for labels
            
            # Add value labels above bars with background
            for bar, rate in zip(bars1, rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{rate:.1%}',
                        ha='center', va='bottom',
                        color='white',
                        fontweight='bold',
                        fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            
            # Calculate average scores
            scores = []
            for data in summary.get("models", {}).values():
                total_rounds = data.get("total_rounds", 0)
                if total_rounds > 0:
                    avg_score = data.get("total_score", 0) / total_rounds
                else:
                    avg_score = 0
                scores.append(avg_score)
            
            # Plot average scores
            bars2 = ax2.bar(models, scores)
            ax2.set_title("Average Score by Model")
            ax2.set_ylabel("Average Score")
            ax2.set_ylim(0, max(scores) + 0.5 if scores else 0.5)  # Increased padding for labels
            
            # Add value labels above bars with background
            for bar, score in zip(bars2, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{score:.2f}',
                        ha='center', va='bottom',
                        color='white',
                        fontweight='bold',
                        fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            
            # Rotate x-axis labels
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error creating charts: {str(e)}")
            traceback.print_exc()

    def process_output(self):
        """Process output from experiment"""
        try:
            while True:
                try:
                    output = self.output_queue.get_nowait()
                    if output == "TERMINATE_EXPERIMENT":
                        # Termination signal received
                        self.experiment_running = False
                        break
                    elif output and hasattr(self, 'output_text'):
                        self.output_text.insert("end", str(output) + "\n")
                        self.output_text.see("end")
                except queue.Empty:
                    break
        finally:
            # Schedule next check
            if self.experiment_running:
                self.window.after(100, self.process_output)
            else:
                self.window.after(1000, self.process_output)

    def validate_config(self, config: Dict[str, Any]):
        """Validate experiment configuration"""
        if config["num_rounds"] < 1:
            raise ValueError("Number of rounds must be positive")
            
        if config["experiment_type"] not in self.configs:
            raise ValueError(f"Invalid experiment type: {config['experiment_type']}")
            
        if not config["models"]:
            raise ValueError("No models selected")

    def stop_experiment(self):
        """Stop the current experiment"""
        if not self.experiment_running:
            messagebox.showwarning("Warning", "No experiment is running")
            return
        
        try:
            # Set flag to stop experiment
            self.experiment_running = False
            
            # Update UI
            if hasattr(self, 'progress_label'):
                self.progress_label.configure(text="Progress: Stopped")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.set(0)
            
            # Add to output log
            if hasattr(self, 'output_text'):
                self.output_text.insert("end", "\nExperiment stopped by user\n")
                self.output_text.see("end")
            
            # Insert termination signal to break infinite loops
            self.output_queue.put("TERMINATE_EXPERIMENT")
            
            # Reset states and UI
            def reset_ui():
                try:
                    if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
                        self.start_btn.configure(state="normal")
                    if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                        self.stop_btn.configure(state="disabled")
                    if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                        self.progress_label.configure(text="Progress: Not Running")
                    if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                        self.progress_bar.set(0)
                except Exception as e:
                    print(f"Error resetting UI: {str(e)}")
            
            # Schedule UI reset
            self.window.after(100, reset_ui)
            
            # Use after to show message to avoid blocking UI
            self.window.after(200, lambda: self.output_queue.put("Experiment stopping..."))
            
        except Exception as e:
            self.output_queue.put(f"Error stopping experiment: {str(e)}")
            traceback.print_exc()

    def create_monitor_panel(self):
        """Create the monitor panel"""
        # Configure grid
        self.monitor_frame.grid_columnconfigure(0, weight=1)
        self.monitor_frame.grid_rowconfigure(1, weight=1)  # Make output log expand
        
        # Title
        monitor_label = ctk.CTkLabel(
            self.monitor_frame,
            text="Experiment Monitor",
            font=("Arial", 12, "bold")
        )
        monitor_label.pack(pady=5)
        
        # Progress section
        progress_frame = ctk.CTkFrame(self.monitor_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Progress: Not Running",
            font=("Arial", 11)
        )
        self.progress_label.pack(pady=2)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=2)
        self.progress_bar.set(0)
        
        # Output log
        log_frame = ctk.CTkFrame(self.monitor_frame)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        log_label = ctk.CTkLabel(
            log_frame,
            text="Output Log:",
            font=("Arial", 11)
        )
        log_label.pack(pady=2)
        
        self.output_text = ctk.CTkTextbox(
            log_frame,
            wrap="word",
            height=200
        )
        self.output_text.pack(fill="both", expand=True, padx=5, pady=2)

    def create_results_panel(self):
        """Create the results panel with tabs"""
        # Create notebook for tabs
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Summary tab (Charts only)
        self.summary_frame = ctk.CTkFrame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")
        
        # Create charts frame that takes full space
        self.charts_frame = ctk.CTkFrame(self.summary_frame)
        self.charts_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Detailed Summary tab
        self.detailed_frame = ctk.CTkFrame(self.results_notebook)
        self.results_notebook.add(self.detailed_frame, text="Detailed Summary")
        
        # Create table in detailed summary tab
        self.table_frame = ctk.CTkFrame(self.detailed_frame)
        self.table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add detailed tree view
        self.detailed_tree = ttk.Treeview(
            self.table_frame,
            columns=("Round", "Model", "Choice", "Partner", "Score", "Cooperation Rate", "Outcome"),
            show="headings",
            height=20
        )
        
        # Configure columns
        for col in ("Round", "Model", "Choice", "Partner", "Score", "Cooperation Rate", "Outcome"):
            self.detailed_tree.heading(col, text=col)
            self.detailed_tree.column(col, width=100)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.detailed_tree.yview)
        x_scrollbar = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.detailed_tree.xview)
        self.detailed_tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Pack tree and scrollbars
        self.detailed_tree.pack(side="left", fill="both", expand=True)
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar.pack(side="bottom", fill="x")
        
        # Initialize summary tree
        self.create_summary_tree()

    def create_summary_tree(self):
        """Create the summary tree"""
        try:
            # Create new summary tree
            self.summary_tree = ttk.Treeview(
                self.table_frame,
                columns=("Model", "Cooperation", "Rounds", "Tokens"),
                show="headings",
                height=5
            )
            
            # Configure columns
            for col in ("Model", "Cooperation", "Rounds", "Tokens"):
                self.summary_tree.heading(col, text=col)
                self.summary_tree.column(col, width=100)
            
            self.summary_tree.pack(fill="x", padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating summary tree: {str(e)}")

    def clear_results(self):
        """Clear all results displays"""
        try:
            # Clear tree
            if hasattr(self, 'summary_tree') and self.summary_tree.winfo_exists():
                for item in self.summary_tree.get_children():
                    self.summary_tree.delete(item)
            
            # Clear charts
            if hasattr(self, 'charts_frame') and self.charts_frame.winfo_exists():
                for widget in self.charts_frame.winfo_children():
                    widget.destroy()
        except Exception as e:
            print(f"Error clearing results: {str(e)}")

    def create_history_viewer(self):
        """Create history viewer window"""
        history_window = ctk.CTkToplevel(self.window)
        history_window.title("Experiment History")
        history_window.geometry("1000x600")
        
        # Create main frame
        main_frame = ctk.CTkFrame(history_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tree view for experiments
        tree_frame = ctk.CTkFrame(main_frame)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        columns = ("Date", "Type", "Condition", "Rounds")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=15
        )
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        tree.pack(fill="both", expand=True, side="left")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Load experiments
        self.load_experiment_history(tree)
        
        # Add refresh button
        refresh_btn = ctk.CTkButton(
            main_frame,
            text="Refresh",
            command=lambda: self.load_experiment_history(tree),
            width=100
        )
        refresh_btn.pack(pady=5)

    def load_experiment_history(self, tree):
        """Load experiment history from database"""
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiments with condition
            cursor.execute("""
                SELECT 
                    timestamp,
                    experiment_type,
                    condition,
                    models,
                    num_rounds,
                    experiment_id
                FROM experiments
                ORDER BY timestamp DESC
            """)
            
            # Add experiments to tree
            for row in cursor.fetchall():
                timestamp, exp_type, condition, models, rounds, exp_id = row
                
                # Convert timestamp to date
                try:
                    date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                except:
                    date = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                # Add to tree
                tree.insert(
                    "",
                    "end",
                    iid=exp_id,
                    values=(date, exp_type, condition or "N/A", models, rounds)
                )
                
            conn.close()
            
        except Exception as e:
            print(f"History loading error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load history: {str(e)}")

    def show_experiment_details(self, exp_id):
        """Show detailed results for selected experiment"""
        try:
            import sqlite3
            
            # Connect to database using env path
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiment details
            cursor.execute("""
                SELECT 
                    results_file,
                    config_file
                FROM experiments
                WHERE experiment_id = ?
            """, (exp_id,))
            
            results_file, config_file = cursor.fetchone()
            conn.close()
            
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            # Create details window
            details_window = ctk.CTkToplevel(self.window)
            details_window.title(f"Experiment Details - {exp_id}")
            details_window.geometry("800x600")
            
            # Create notebook for tabs
            notebook = ttk.Notebook(details_window)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_frame = ctk.CTkFrame(notebook)
            notebook.add(summary_frame, text="Summary")
            
            # Add summary information
            summary_text = ctk.CTkTextbox(summary_frame, wrap="word")
            summary_text.pack(fill="both", expand=True, padx=5, pady=5)
            
            summary = f"""Experiment ID: {exp_id}
Results File: {results_file}
Config File: {config_file}

Summary Statistics:
-----------------
"""
            
            for model, data in results.get("summary", {}).items():
                summary += f"\nModel: {model}"
                summary += f"\nCooperation Rate: {data.get('cooperation_rate', 0):.2%}"
                summary += f"\nTotal Rounds: {data.get('total_rounds', 0)}"
                summary += f"\nToken Usage: {data.get('token_usage', {}).get('total_tokens', 0)}"
                summary += "\n-----------------"
                
            summary_text.insert("1.0", summary)
            
            # Charts tab
            charts_frame = ctk.CTkFrame(notebook)
            notebook.add(charts_frame, text="Charts")
            
            # Create charts using results data
            self.create_result_charts(charts_frame, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load experiment details: {str(e)}")

    def run(self):
        """Start the application"""
        self.window.mainloop()

    def load_history(self):
        """Load and display experiment history"""
        try:
            # Clear existing items first
            self.main_tree.delete(*self.main_tree.get_children())
            self.test_tree.delete(*self.test_tree.get_children())
            
            # Load results from results directory
            results_dir = Path("results")
            if not results_dir.exists():
                print("No results directory found")
                return
            
            # Process test runs
            test_runs_dir = results_dir / "test_runs"
            if test_runs_dir.exists():
                self.process_test_results(test_runs_dir)
            
            # Process main runs
            main_runs_dir = results_dir / "main_runs"
            if main_runs_dir.exists():
                self.process_main_results(main_runs_dir)
            
            # Update the display
            self.update_history()
            
        except Exception as e:
            print(f"Failed to load history: {str(e)}")

    def create_result_charts(self, frame, results):
        """Create visualization of experiment results"""
        try:
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            plt.style.use('dark_background')  # Use dark theme
            
            # Extract data
            models = []
            coop_rates = []
            avg_scores = []
            
            for model_name, model_data in results.items():
                if isinstance(model_data, dict) and "rounds" in model_data:
                    models.append(model_name)
                    
                    # Calculate cooperation rate
                    total_rounds = len(model_data["rounds"])
                    coop_choices = sum(1 for round in model_data["rounds"] if round["player_choice"] == "A")
                    coop_rate = (coop_choices / total_rounds) * 100 if total_rounds > 0 else 0
                    coop_rates.append(coop_rate)
                    
                    # Calculate average score
                    total_score = sum(round["player_score"] for round in model_data["rounds"])
                    avg_score = total_score / total_rounds if total_rounds > 0 else 0
                    avg_scores.append(avg_score)
            
            if not models:
                ctk.CTkLabel(frame, text="No data available for visualization").pack(pady=20)
                return
            
            # Plot cooperation rates
            bars1 = ax1.bar(models, coop_rates, color='lightblue')
            ax1.set_title('Cooperation Rate by Model')
            ax1.set_ylabel('Cooperation Rate (%)')
            ax1.set_ylim(0, max(coop_rates) + 15)  # Increased padding for labels
            
            # Add value labels above bars with background
            for bar, rate in zip(bars1, coop_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{rate:.1f}%',
                        ha='center', va='bottom',
                        color='white',
                        fontweight='bold',
                        fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            
            # Plot average scores
            bars2 = ax2.bar(models, avg_scores, color='lightgreen')
            ax2.set_title('Average Score by Model')
            ax2.set_ylabel('Average Score')
            ax2.set_ylim(0, max(avg_scores) + 0.8)  # Increased padding for labels
            
            # Add value labels above bars with background
            for bar, score in zip(bars2, avg_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                        f'{score:.2f}',
                        ha='center', va='bottom',
                        color='white',
                        fontweight='bold',
                        fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            
            # Rotate x-axis labels
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Clear existing widgets
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error creating charts: {str(e)}")
            traceback.print_exc()

    def on_experiment_type_change(self, _):
        """Handle experiment type change"""
        self.update_condition_options()

    def update_condition_options(self):
        """Update condition options based on selected experiment type"""
        exp_type = self.exp_type.get()
        if exp_type in self.configs:
            # Get conditions from config
            conditions = self.configs[exp_type].get("conditions", {})
            if isinstance(conditions, dict):
                # If conditions is a dictionary, use the condition types
                condition_options = list(conditions.keys())
            else:
                # Fallback to list or default
                condition_options = conditions if isinstance(conditions, list) else ["neutral"]
            
            # Update dropdown values
            self.condition_dropdown.configure(values=condition_options)
            
            # Set default condition
            if condition_options and self.condition_var.get() not in condition_options:
                self.condition_var.set(condition_options[0])

    def create_history_panel(self):
        """Create experiment history panel"""
        ctk.CTkLabel(self.history_frame, text="Experiment History:").pack(anchor="w", padx=5, pady=5)
        
        # Create container frame
        container = ctk.CTkFrame(self.history_frame)
        container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create tree view
        columns = ("Date", "Type", "Condition", "Rounds", "ID")
        self.history_tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            height=10
        )
        
        # Configure columns
        for col in columns[:4]:  # Only show first 4 columns, hide the ID column
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        # Hide the ID column
        self.history_tree.heading("ID", text="ID")
        self.history_tree.column("ID", width=0, stretch=False)
        
        # Add custom scrollbar
        scrollbar = ctk.CTkScrollbar(container, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.history_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # After creating self.history_tree in create_history_panel, bind the selection event
        self.history_tree.bind('<<TreeviewSelect>>', self.on_history_select)

    def on_history_select(self, event):
        """Handle history item selection"""
        try:
            # Get selected item
            selected_item = self.history_tree.selection()
            if not selected_item:
                return
            
            # Get experiment ID directly - it's both the tree item ID and the last column
            experiment_id = selected_item[0]  # The iid is the experiment_id
            
            print(f"Selected experiment ID: {experiment_id}")
            
            # Load data for this experiment directly by ID
            results = self.load_experiment_by_id(experiment_id)
            
            # Update UI with results
            self.output_text.insert("end", f"\nLoaded experiment results: {experiment_id}\n")
            self.output_text.see("end")
            
            # Update results displays
            self.update_results_display(results)
            self.update_detailed_summary(results)
            
            # Switch to Summary tab
            self.results_notebook.select(0)  # Select the first tab (Summary)
            
        except Exception as e:
            print(f"Error loading experiment results: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load experiment results: {str(e)}")

    def load_experiment_by_id(self, experiment_id):
        """Load experiment data directly by experiment ID"""
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiment details using experiment_id
            cursor.execute("""
                SELECT experiment_id, timestamp, experiment_type, condition, results_file, config_file
                FROM experiments
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment results not found for ID: {experiment_id}")
            
            exp_id, timestamp, exp_type, condition, results_file, config_file = row
            
            print(f"Loading experiment: ID={exp_id}, Type={exp_type}, Condition={condition}")
            print(f"Results file path: {results_file}")
            
            # Check if the combined results file exists
            if results_file and Path(results_file).exists():
                print(f"Loading combined results from: {results_file}")
                with open(results_file, 'r') as f:
                    combined_data = json.load(f)
                    
                # Extract model results from combined data
                if "results" in combined_data and isinstance(combined_data["results"], dict):
                    all_results = combined_data["results"]
                    print(f"Loaded results for {len(all_results)} models from combined file")
                    
                    # Store current results
                    self.current_results = all_results
                    return all_results
            
            # If combined file not found or doesn't have results, try alternative methods
            # First check if results_file exists and is valid
            results_dir = Path(results_file).parent if results_file else None
            
            if not results_dir or not results_dir.exists():
                # Use experiment_id to construct the directory path
                results_dir = Path("results/experiments") / experiment_id
                
            if not results_dir.exists():
                raise ValueError(f"Results directory not found: {results_dir}")
            
            print(f"Using results directory: {results_dir}")
            
            # Load results from all model files in the directory
            all_results = {}
            
            # Look for model_results.json files
            model_files = list(results_dir.glob("*_results.json"))
            print(f"Found {len(model_files)} model result files")
            
            for result_file in model_files:
                try:
                    print(f"Loading file: {result_file}")
                    with open(result_file) as f:
                        model_result = json.load(f)
                        
                        # Extract model name - either from the file name or the content
                        if "model" in model_result:
                            model_name = model_result["model"]
                        else:
                            # Try to extract from filename (e.g., gemini-1.5-pro_results.json)
                            file_name = result_file.stem
                            model_name = file_name.replace("_results", "")
                        
                        print(f"Loaded data for model: {model_name}")
                        all_results[model_name] = model_result
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
                    traceback.print_exc()
            
            # If no individual model results, try combined results file
            if not all_results:
                combined_file = results_dir / "combined_results.json"
                if combined_file.exists():
                    print(f"Loading combined results file: {combined_file}")
                    with open(combined_file) as f:
                        combined_data = json.load(f)
                        if isinstance(combined_data, dict):
                            # Extract model data
                            for key, value in combined_data.items():
                                if key not in ["timestamp", "experiment_id", "config_file", "is_test", "results_dir"] and isinstance(value, dict):
                                    all_results[key] = value
                                    print(f"Extracted model data for: {key}")
            
            if not all_results:
                raise ValueError(f"No valid result files found in experiment directory: {results_dir}")
            
            print(f"Successfully loaded data for {len(all_results)} models")
            
            # Store current results
            self.current_results = all_results
            
            return all_results
            
        except Exception as e:
            print(f"Error loading experiment data: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def update_history(self):
        """Update experiment history display"""
        if not hasattr(self, 'history_tree') or not self.history_tree.winfo_exists():
            print("History tree not available for update")
            return
            
        try:
            # Clear existing items
            self.history_tree.delete(*self.history_tree.get_children())
                
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiments with condition
            cursor.execute("""
                SELECT 
                    experiment_id,
                    timestamp,
                    experiment_type,
                    condition,
                    num_rounds
                FROM experiments 
                ORDER BY timestamp DESC
            """)
            
            # Add experiments to tree
            rows = cursor.fetchall()
            for row in rows:
                experiment_id, timestamp, exp_type, condition, rounds = row
                
                # Convert timestamp to date string
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                
                # Add to tree - use experiment_id as the iid (tree identifier)
                self.history_tree.insert(
                    "",
                    "end",
                    iid=experiment_id,  # Use the experiment_id as the unique identifier
                    values=(
                        date_str,
                        exp_type,
                        condition or "neutral",  # Default to neutral if None
                        rounds,
                        experiment_id  # Include experiment_id as the last column (hidden)
                    )
                )
            
            print(f"Updated history: {len(rows)} experiments")
            self.history_loaded = True
                
        except Exception as e:
            print(f"Error updating history: {str(e)}")
            traceback.print_exc()
        finally:
            if 'conn' in locals():
                conn.close()

    def update_detailed_summary(self, results):
        """Update the detailed summary tab with experiment results"""
        try:
            # Clear existing items
            self.detailed_tree.delete(*self.detailed_tree.get_children())
            
            # Add detailed round data
            for model_name, model_data in results.items():
                if isinstance(model_data, dict) and "rounds" in model_data:
                    rounds_data = model_data["rounds"]
                    
                    for round_data in rounds_data:
                        round_num = round_data.get("round_number", "N/A")
                        player_choice = round_data.get("player_choice", "N/A")
                        partner_choice = round_data.get("partner_choice", "N/A")
                        player_score = round_data.get("player_score", "N/A")
                        
                        # Calculate cooperation rate up to this round
                        if "cumulative_player_coop_rate_pre" in round_data:
                            coop_rate = f"{round_data['cumulative_player_coop_rate_pre']:.2%}"
                        else:
                            coop_rate = "N/A"
                            
                        # Get outcome
                        outcome = round_data.get("outcome_lag1", "N/A")
                        
                        # Add to tree
                        self.detailed_tree.insert(
                            "",
                            "end",
                            values=(
                                round_num,
                                model_name,
                                player_choice,
                                partner_choice,
                                player_score,
                                coop_rate,
                                outcome
                            )
                        )
                        
        except Exception as e:
            print(f"Error updating detailed summary: {str(e)}")
            traceback.print_exc()

    def initial_load_history(self):
        """Initial loading of experiment history at application startup"""
        print("Attempting initial history load...")
        
        # Check if the database exists
        if not os.path.exists(self.db_path):
            print(f"Database file not found at {self.db_path}")
            return
        
        # Make sure all UI components are ready
        if not hasattr(self, 'history_tree') or not self.history_tree.winfo_exists():
            print("History tree not ready yet, retrying in 1 second...")
            self.window.after(1000, self.initial_load_history)
            return
        
        # Attempt to load history
        try:
            self.history_tree.delete(*self.history_tree.get_children())
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiments with condition
            cursor.execute("""
                SELECT 
                    experiment_id,
                    timestamp,
                    experiment_type,
                    condition,
                    num_rounds
                FROM experiments 
                ORDER BY timestamp DESC
                LIMIT 50  -- Limit to 50 most recent experiments
            """)
            
            # Add experiments to tree
            rows = cursor.fetchall()
            if not rows:
                print("No experiment history found in database")
            else:
                for row in rows:
                    experiment_id, timestamp, exp_type, condition, rounds = row
                    
                    # Convert timestamp to date string
                    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                    
                    # Add to tree - use experiment_id as the iid (tree identifier)
                    self.history_tree.insert(
                        "",
                        "end",
                        iid=experiment_id,  # Use the experiment_id as the unique identifier
                        values=(
                            date_str,
                            exp_type,
                            condition or "neutral",  # Default to neutral if None
                            rounds,
                            experiment_id  # Include experiment_id as the last column (hidden)
                        )
                    )
                
                print(f"Successfully loaded {len(rows)} experiments into history")
                self.history_loaded = True
            
            conn.close()
            
        except Exception as e:
            print(f"Error in initial history loading: {str(e)}")
            traceback.print_exc()
            
            # Try again after a delay
            if not self.history_loaded:
                print("Retrying history load in 2 seconds...")
                self.window.after(2000, self.initial_load_history)

if __name__ == "__main__":
    app = LLMExperimentApp()
    app.run() 