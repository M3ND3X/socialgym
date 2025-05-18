import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import glob

class TestDataExporter:
    def __init__(self, experiment_dir: str):
        """Initialize with experiment output directory"""
        self.experiment_dir = Path(experiment_dir)
        self.results_files = [f for f in self.experiment_dir.glob("*_results.json") 
                             if f.name != "experiment_results.json"]
        
    def export_to_excel(self, output_path: str):
        """Export all test data to Excel file with multiple sheets"""
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Export different aspects of the data
            self._export_summary_sheet(writer)
            self._export_choices_sheet(writer)
            self._export_reasoning_sheet(writer)
            self._export_model_performance_sheet(writer)
            self._export_round_details_sheet(writer)
    
    def _export_summary_sheet(self, writer):
        """Export overall experiment summary"""
        summary_data = []
        
        for result_file in self.results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            model_name = data.get("model", "Unknown")
            summary = {
                "Model": model_name,
                "Total Rounds": data["summary"]["total_rounds"],
                "Option A Rate": (data["summary"]["choices"]["A"] / data["summary"]["total_rounds"]) * 100,
                "Option B Rate": (data["summary"]["choices"]["B"] / data["summary"]["total_rounds"]) * 100,
                "Experiment Type": data["config"]["experiment_type"],
                "Condition": data["config"].get("condition", "N/A")
            }
            summary_data.append(summary)
            
        df_summary = pd.DataFrame(summary_data)
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    def _export_choices_sheet(self, writer):
        """Export detailed choice analysis"""
        choices_data = []
        
        for result_file in self.results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            model_name = data.get("model", "Unknown")
            for round_data in data["rounds"]:
                choice_data = {
                    "Model": model_name,
                    "Round": round_data["round_number"],
                    "Player Choice": round_data["player_choice"],
                    "Partner Choice": round_data["partner_choice"],
                    "Player Score": round_data["player_score"],
                    "Partner Score": round_data["partner_score"]
                }
                choices_data.append(choice_data)
                
        df_choices = pd.DataFrame(choices_data)
        if not df_choices.empty:
            df_choices.to_excel(writer, sheet_name='Choices', index=False)
    
    def _export_reasoning_sheet(self, writer):
        """Export reasoning analysis"""
        reasoning_data = []
        
        for result_file in self.results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            model_name = data.get("model", "Unknown")
            for round_data in data["rounds"]:
                reasoning = {
                    "Model": model_name,
                    "Round": round_data["round_number"],
                    "Choice": round_data["player_choice"],
                    "Reasoning": round_data["player_reasoning"]
                }
                reasoning_data.append(reasoning)
                
        df_reasoning = pd.DataFrame(reasoning_data)
        if not df_reasoning.empty:
            df_reasoning.to_excel(writer, sheet_name='Reasoning', index=False)
    
    def _export_model_performance_sheet(self, writer):
        """Export model performance metrics"""
        performance_data = []
        
        for result_file in self.results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            model_name = data.get("model", "Unknown")
            token_usage = data["summary"]["token_usage"]
            
            perf = {
                "Model": model_name,
                "Total Tokens": token_usage["total_tokens"],
                "Prompt Tokens": token_usage["prompt_tokens"],
                "Completion Tokens": token_usage["completion_tokens"]
            }
            performance_data.append(perf)
            
        df_performance = pd.DataFrame(performance_data)
        if not df_performance.empty:
            df_performance.to_excel(writer, sheet_name='Model Performance', index=False)
    
    def _export_round_details_sheet(self, writer):
        """Export detailed round information"""
        round_data = []
        
        for result_file in self.results_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            model_name = data.get("model", "Unknown")
            for round_info in data["rounds"]:
                round_detail = {
                    "Model": model_name,
                    "Round": round_info["round_number"],
                    "Player Choice": round_info["player_choice"],
                    "Partner Choice": round_info["partner_choice"],
                    "System Prompt": round_info["system_prompt"],
                    "User Prompt": round_info["user_prompt"],
                    "Raw Response": round_info["raw_model_response"]
                }
                round_data.append(round_detail)
                
        df_rounds = pd.DataFrame(round_data)
        if not df_rounds.empty:
            df_rounds.to_excel(writer, sheet_name='Round Details', index=False)

def main():
    """Main function to run the export"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export test results to Excel")
    parser.add_argument("--ex", required=True, help="Experiment output directory")
    args = parser.parse_args()
    
    # Get the experiment directory
    results_dir = Path("results/experiments")
    experiment_dir = results_dir / args.ex
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory {experiment_dir} not found")
        return
        
    # Create output filename
    output_file = f"analysis_{args.ex}.xlsx"
    
    # Export the data
    exporter = TestDataExporter(str(experiment_dir))
    exporter.export_to_excel(output_file)
    print(f"Results exported to {output_file}")

if __name__ == "__main__":
    main() 