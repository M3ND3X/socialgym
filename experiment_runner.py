from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
import time
from pathlib import Path
import re

from experiment_management import ExperimentManager
from prompt_generation import PromptGenerator
from model_interface import ModelInterface
from game_history import GameHistory

class ExperimentRunner:
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        num_rounds: int = 1,
        results_dir: str = None,
        condition_type: str = "neutral",
        memory_type: str = None
    ):
        self.model_name = model_name
        self.config = config
        self.num_rounds = num_rounds
        self.results_dir = results_dir or "results/test_runs"
        self.condition_type = condition_type
        self.memory_type = memory_type
        
        # Ensure results directory exists
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.prompt_generator = PromptGenerator()
        self.experiment_manager = ExperimentManager()
        self.game_history = GameHistory()
        
        # Setup model interface with config
        self._setup_model_interface()
        
    def _setup_model_interface(self):
        """Set up model interface with appropriate settings"""
        # Load model config
        with open("config/model_config.json", 'r') as f:
            model_config = json.load(f)
            
        # Get model settings
        if self.model_name not in model_config["models"]:
            raise ValueError(f"Model {self.model_name} not found in config")
        
        model_settings = model_config["models"][self.model_name]
        
        # Initialize model interface with settings
        self.model_interface = ModelInterface(
            model_name=self.model_name,
            temperature=model_settings.get("temperature", 0.7),
            top_p=model_settings.get("top_p", 0.95),
            top_k=model_settings.get("top_k", 40),
            max_output_tokens=model_settings.get("max_output_tokens", 1024),
            family=model_settings.get("family", "gemini")
        )
        
    def _get_memory_prompt(self) -> str:
        """Generate memory prompt based on game history and settings"""
        if not self.game_history.rounds:
            return ""
        
        memory_settings = self.config.get("memory_settings", {})
        if not memory_settings.get("enabled", False):
            return ""
        
        # Get last round
        last_round = self.game_history.rounds[-1]
        
        # Format the outcome
        outcome = "cooperation" if (last_round.player_choice.value == "A" and last_round.partner_choice.value == "A") else "defection"
        points = f"{last_round.player_score} points each" if last_round.player_score == last_round.partner_score else f"{last_round.player_score} vs {last_round.partner_score} points"
        
        return f"In the previous round:\nYou chose {last_round.player_choice.value}, Partner chose {last_round.partner_choice.value}\nOutcome: {outcome} ({points})"
        
    def run_experiment(self) -> Dict[str, Any]:
        """Run experiment and return results
        
        The results include feedback variables for each round:
        - player_choice_lag1: The player's choice ('A' or 'B') from the previous round (t-1)
        - partner_choice_lag1: The partner's choice ('A' or 'B') from the previous round (t-1)
        - player_score_lag1: The player's score from the previous round (t-1)
        - partner_score_lag1: The partner's score from the previous round (t-1)
        - outcome_lag1: Categorical description of the previous round's outcome
        - cumulative_player_coop_rate_pre: Proportion of 'A' choices by player in all previous rounds
        - cumulative_partner_coop_rate_pre: Proportion of 'A' choices by partner in all previous rounds
        - cumulative_player_score_pre: Sum of the player's scores from all previous rounds
        
        All variables reflect the state before the current round t (based on rounds 1 to t-1).
        """
        # Get the results directory from the instance
        experiment_dir = Path(self.results_dir)
        
        try:
            # Get condition-specific prompt template
            if "conditions" not in self.config:
                prompt_template = self.config.get("prompt_template", "")
            else:
                conditions = self.config["conditions"]
                condition = conditions.get(self.condition_type, conditions.get("neutral", {}))
                prompt_template = condition.get("prompt_template", "")
            
            rounds = []
            memory = []  # Initialize memory list
            
            for round_num in range(1, self.num_rounds + 1):
                max_retries = 3
                retry_count = 0
                round_success = False

                while not round_success:
                    try:
                        # Generate prompts
                        system_prompt = self.prompt_generator.generate_system_prompt(self.config)
                        user_prompt = prompt_template
                        
                        # Add memory based on memory type
                        if memory:
                            memory_prompt = "\nPrevious rounds:\n"
                            if self.memory_type == "short_term_memory":
                                # Only show last round for short-term memory
                                last_round = memory[-1]
                                memory_prompt += (
                                    f"Round {last_round['round']}: "
                                    f"You chose {last_round['player_choice']}, "
                                    f"Partner chose {last_round['partner_choice']} - "
                                    f"You got {last_round['player_score']} points, "
                                    f"Partner got {last_round['partner_score']} points\n"
                                )
                            elif self.memory_type == "long_term_memory":
                                # Show all previous rounds for long-term memory
                                for prev_round in memory:
                                    memory_prompt += (
                                        f"Round {prev_round['round']}: "
                                        f"You chose {prev_round['player_choice']}, "
                                        f"Partner chose {prev_round['partner_choice']} - "
                                        f"You got {prev_round['player_score']} points, "
                                        f"Partner got {prev_round['partner_score']} points\n"
                                    )
                            
                            # Insert memory after condition description
                            prompt_parts = user_prompt.split('\n\n', 1)
                            if len(prompt_parts) > 1:
                                user_prompt = f"{prompt_parts[0]}{memory_prompt}\n{prompt_parts[1]}"
                            else:
                                user_prompt = f"{user_prompt}{memory_prompt}"
                        
                        # Get model response
                        response = self.model_interface.generate_response(system_prompt, user_prompt)
                        
                        try:
                            # Extract choice and reasoning
                            choice, reasoning = self._extract_choice_and_reasoning(response["raw_response"])
                            round_success = True  # If successful, mark as successful
                        except ValueError as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                print(f"\nRetrying round {round_num} ({retry_count}/{max_retries})")
                                continue
                            else:
                                print(f"\nMax retries reached for round {round_num}. Waiting 30 seconds before trying again...")
                                time.sleep(30)  # Wait 30 seconds
                                retry_count = 0  # Reset retry counter
                                continue  # Try again

                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"\nError in round {round_num}: {str(e)}")
                            print(f"Retrying ({retry_count}/{max_retries})...")
                            continue
                        else:
                            print(f"\nMax retries reached for round {round_num} due to error: {str(e)}")
                            print(f"Waiting 30 seconds before trying again...")
                            time.sleep(30)  # Wait 30 seconds
                            retry_count = 0  # Reset retry counter
                            continue  # Try again

                # Rest of the round processing
                partner_choice = self._simulate_partner_choice(self.game_history)
                
                # Calculate scores
                player_score = self._calculate_score(choice, partner_choice)
                partner_score = self._calculate_score(partner_choice, choice)
                
                # Calculate feedback variables before updating game history
                # Check if it's the first round
                is_first_round = len(self.game_history.rounds) == 0
                
                # Initialize feedback variables with defaults for first round
                player_choice_lag1 = None
                partner_choice_lag1 = None
                player_score_lag1 = None
                partner_score_lag1 = None
                outcome_lag1 = None
                cumulative_player_coop_rate_pre = 0.0
                cumulative_partner_coop_rate_pre = 0.0
                cumulative_player_score_pre = 0
                
                if not is_first_round:
                    # Get data from the previous round (t-1)
                    last_round = self.game_history.rounds[-1]
                    player_choice_lag1 = last_round.player_choice.value
                    partner_choice_lag1 = last_round.partner_choice.value
                    player_score_lag1 = last_round.player_score
                    partner_score_lag1 = last_round.partner_score
                    
                    # Calculate outcome_lag1 based on choices in the previous round
                    outcome_lag1 = self.game_history.get_outcome_type(player_choice_lag1, partner_choice_lag1)
                    
                    # Calculate cumulative stats based on all previous rounds
                    num_prior_rounds = len(self.game_history.rounds)
                    
                    # Cooperation rates
                    cumulative_player_coop_rate_pre = self.game_history.get_cooperation_rate("player")
                    cumulative_partner_coop_rate_pre = self.game_history.get_cooperation_rate("partner")
                    
                    # Cumulative player score
                    cumulative_player_score_pre = self.game_history.get_cumulative_score("player")
                
                # Update memory for both memory types
                memory.append({
                    'round': round_num,
                    'player_choice': choice,
                    'partner_choice': partner_choice,
                    'player_score': player_score,
                    'partner_score': partner_score
                })
                
                # Record round
                round_data = {
                    "round_number": round_num,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_model_response": response["raw_response"],
                    "player_choice": choice,
                    "player_reasoning": reasoning,
                    "partner_choice": partner_choice,
                    "player_score": player_score,
                    "partner_score": partner_score,
                    # Add the new feedback variables
                    "player_choice_lag1": player_choice_lag1,
                    "partner_choice_lag1": partner_choice_lag1,
                    "player_score_lag1": player_score_lag1, 
                    "partner_score_lag1": partner_score_lag1,
                    "outcome_lag1": outcome_lag1,
                    "cumulative_player_coop_rate_pre": cumulative_player_coop_rate_pre,
                    "cumulative_partner_coop_rate_pre": cumulative_partner_coop_rate_pre,
                    "cumulative_player_score_pre": cumulative_player_score_pre,
                    "game_history": self.game_history.get_formatted_history(),
                    "token_usage": response.get("token_usage", {}),
                    "total_tokens_used": response.get("total_tokens_used", 0)
                }
                
                rounds.append(round_data)
                
                # Update game history
                self.game_history.add_round(choice, partner_choice, round_num)
                
            # Prepare results
            results = {
                "model": self.model_name,
                "config": self.config,
                "condition": self.condition_type,  # Add condition to results
                "rounds": rounds,
                "summary": {
                    "total_rounds": len(rounds),
                    "choices": {
                        "A": sum(1 for r in rounds if r["player_choice"] == "A"),
                        "B": sum(1 for r in rounds if r["player_choice"] == "B")
                    },
                    "token_usage": self.model_interface.get_token_usage()
                }
            }
            
            # We no longer save individual model results here, because main.py handles this
            return results
            
        except Exception as e:
            print(f"Error running experiment: {str(e)}")
            raise


    def _extract_choice_and_reasoning(self, response: str) -> tuple[str, str]:
        """Extract choice and reasoning from model response"""
        try:
            # First look for explicit choice statements
            choice_patterns = [
                # Choice section patterns with quotes
                r"(?i)\*\*choice:\*\*\s*\n?\s*'option\s*([AB])'",
                r"(?i)\*\*choice:\*\*\s*\n?\s*\"option\s*([AB])\"",
                # Choice section patterns without quotes
                r"(?i)\*\*choice:\*\*\s*\n\s*(?:option\s+)?([AB])\b",
                r"(?i)\*\*choice:\*\*\s*(?:option\s+)?([AB])\b",
                r"(?i)choice:\s*(?:option\s+)?([AB])\b",
                # Quoted option patterns
                r"(?i)(?:choice|choose)(?:\s*:|:?\s+)'option\s*([AB])'",
                r"(?i)(?:choice|choose)(?:\s*:|:?\s+)\"option\s*([AB])\"",
                # Bold option patterns
                r"(?i)(?:choice|choose)(?:\s*:|:?\s+)\*\*option\s*([AB])\*\*",
                # Plain option patterns
                r"(?i)(?:choice|choose)(?:\s*:|:?\s+)option\s*([AB])\b",
                # Therefore patterns
                r"(?i)therefore,?\s+(?:i\s+)?choose\s*\*\*option\s*([AB])\*\*",
                r"(?i)therefore,?\s+(?:i\s+)?choose[:\s]+option\s*([AB])\b",
                # "Is to" patterns
                r"(?i)(?:is|would be)\s+to\s+\*\*option\s*([AB])\*\*",
                r"(?i)(?:is|would be)\s+to\s+option\s*([AB])\b",
                # Simple option patterns at end
                r"(?i)'option\s*([AB])'(?:\s*$|\s*\n)",
                r"(?i)\"option\s*([AB])\"(?:\s*$|\s*\n)", 
                r"(?i)\*\*option\s*([AB])\*\*(?:\s*$|\s*\n)",
                r"(?i)option\s*([AB])(?:\s*$|\s*\n)",
                # Fallback patterns for any option mention
                r"(?i)\*\*option\s*([AB])\*\*",
                r"(?i)\boption\s*([AB])\b"
            ]
            
            # Try patterns on stripped response
            response = response.strip()
            for pattern in choice_patterns:
                match = re.search(pattern, response)
                if match:
                    choice = match.group(1).upper()
                    # Get everything before this choice mention as reasoning
                    reasoning = response[:match.start()].strip()
                    if not reasoning:
                        # If no reasoning before, get everything after
                        reasoning = response[match.end():].strip()
                    return choice, reasoning

            # If no choice found, print response and raise error
            print("\nRaw model response (no choice found):")
            print(response)
            raise ValueError("Could not parse choice from response")

        except Exception as e:
            raise ValueError(f"Failed to parse choice: {str(e)}")



    def _simulate_partner_choice(self, history: GameHistory) -> str:
        """Simulate partner's choice based on history"""
        import random
        
        if not history.rounds:
            # Start with higher chance of defection
            return "B" if random.random() < 0.7 else "A"
        
        # Get last round
        last_round = history.rounds[-1]
        
        if last_round.player_choice.value == "A":
            # If player cooperated, only 30% chance to cooperate back
            return "A" if random.random() < 0.3 else "B"
        else:
            # If player defected, only 10% chance to cooperate
            return "A" if random.random() < 0.1 else "B"

    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results"""
        try:
            # Initialize experiment results
            results = {
                "model": self.model_name,
                "config": self.config,
                "rounds": [],
                "status": "success",
                "summary": {
                    "total_rounds": self.config["num_rounds"],
                    "memory_type": self.config["memory_type"]
                }
            }
            
            # Run experiment rounds
            for round_num in range(self.config["num_rounds"]):
                round_result = self.run_round(round_num)
                results["rounds"].append(round_result)
            
            return results
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            # Return error status if experiment fails
            return {
                "status": "error",
                "error": str(e),
                "model": self.model_name,
                "config": self.config
            }

    def _get_prompt_template(self) -> str:
        """Get the appropriate prompt template based on condition"""
        conditions = self.config.get("conditions", {})
        condition = conditions.get(self.condition_type, conditions.get("neutral", {}))
        return condition.get("prompt_template", "")

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """Run a single round of the experiment
        
        Feedback variables:
        - player_choice_lag1: The player's choice ('A' or 'B') from the previous round (t-1)
        - partner_choice_lag1: The partner's choice ('A' or 'B') from the previous round (t-1)
        - player_score_lag1: The player's score from the previous round (t-1)
        - partner_score_lag1: The partner's score from the previous round (t-1)
        - outcome_lag1: Categorical description of the previous round's outcome
        - cumulative_player_coop_rate_pre: Proportion of 'A' choices by player in all previous rounds
        - cumulative_partner_coop_rate_pre: Proportion of 'A' choices by partner in all previous rounds
        - cumulative_player_score_pre: Sum of the player's scores from all previous rounds
        
        All variables reflect the state before the current round t (based on rounds 1 to t-1).
        """
        max_retries = 3
        retry_count = 0
        round_success = False

        while not round_success:
            try:
                # Get system and user prompts
                system_prompt = "You are participating in a competitive decision-making game..."
                user_prompt = self._generate_prompt(round_num)
                
                # Get model response
                response = self.model_interface.generate_response(system_prompt, user_prompt)
                
                try:
                    # Try to extract choice and reasoning
                    player_choice, player_reasoning = self._extract_choice_and_reasoning(response["raw_response"])
                    round_success = True  # Successfully extracted choice
                except ValueError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Failed to parse choice, retrying round {round_num} ({retry_count}/{max_retries})")
                        continue
                    else:
                        print(f"Max retries reached for round {round_num}. Waiting 30 seconds before trying again...")
                        time.sleep(30)  # Wait 30 seconds
                        retry_count = 0  # Reset retry counter
                        continue  # Try again
                
                # Simulate partner's choice
                partner_choice = self._simulate_partner_choice(self.game_history)
                
                # Calculate scores
                player_score = self._calculate_score(player_choice, partner_choice)
                partner_score = self._calculate_score(partner_choice, player_choice)
                
                # Calculate feedback variables before updating game history
                # Check if it's the first round
                is_first_round = len(self.game_history.rounds) == 0
                
                # Initialize feedback variables with defaults for first round
                player_choice_lag1 = None
                partner_choice_lag1 = None
                player_score_lag1 = None
                partner_score_lag1 = None
                outcome_lag1 = None
                cumulative_player_coop_rate_pre = 0.0
                cumulative_partner_coop_rate_pre = 0.0
                cumulative_player_score_pre = 0
                
                if not is_first_round:
                    # Get data from the previous round (t-1)
                    last_round = self.game_history.rounds[-1]
                    player_choice_lag1 = last_round.player_choice.value
                    partner_choice_lag1 = last_round.partner_choice.value
                    player_score_lag1 = last_round.player_score
                    partner_score_lag1 = last_round.partner_score
                    
                    # Calculate outcome_lag1 based on choices in the previous round
                    outcome_lag1 = self.game_history.get_outcome_type(player_choice_lag1, partner_choice_lag1)
                    
                    # Calculate cumulative stats based on all previous rounds
                    num_prior_rounds = len(self.game_history.rounds)
                    
                    # Cooperation rates
                    cumulative_player_coop_rate_pre = self.game_history.get_cooperation_rate("player")
                    cumulative_partner_coop_rate_pre = self.game_history.get_cooperation_rate("partner")
                    
                    # Cumulative player score
                    cumulative_player_score_pre = self.game_history.get_cumulative_score("player")
                
                # Update game history *after* calculating feedback variables
                self.game_history.add_round(player_choice, partner_choice, round_num + 1)
                
                # Prepare the round_data dictionary with feedback variables
                round_data = {
                    "round_number": round_num + 1,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_model_response": response["raw_response"],
                    "player_choice": player_choice,
                    "player_reasoning": player_reasoning,
                    "partner_choice": partner_choice,
                    "player_score": player_score,
                    "partner_score": partner_score,
                    # Add the new feedback variables
                    "player_choice_lag1": player_choice_lag1,
                    "partner_choice_lag1": partner_choice_lag1,
                    "player_score_lag1": player_score_lag1, 
                    "partner_score_lag1": partner_score_lag1,
                    "outcome_lag1": outcome_lag1,
                    "cumulative_player_coop_rate_pre": cumulative_player_coop_rate_pre,
                    "cumulative_partner_coop_rate_pre": cumulative_partner_coop_rate_pre,
                    "cumulative_player_score_pre": cumulative_player_score_pre,
                    "game_history": self.game_history.get_formatted_history(),
                    "token_usage": response["token_usage"],
                    "total_tokens_used": response["total_tokens_used"]
                }
                
                return round_data
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error in round {round_num}, retrying ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(1)  # Add small delay between retries
                    continue
                else:
                    print(f"Max retries reached for round {round_num} due to error: {str(e)}")
                    print(f"Waiting 30 seconds before trying again...")
                    time.sleep(30)  # Wait 30 seconds
                    retry_count = 0  # Reset retry counter
                    continue  # Try again

        # This line should never be reached with the new infinite retry logic
        raise ValueError(f"Round {round_num} failed after maximum retries")


    def _calculate_score(self, player_choice: str, partner_choice: str) -> int:
        """Calculate score based on choices"""
        payoff_matrix = self.config.get("payoff_matrix", {
            "cooperate_cooperate": [3, 3],
            "cooperate_defect": [0, 5],
            "defect_cooperate": [5, 0],
            "defect_defect": [1, 1]
        })
        
        if player_choice == "A" and partner_choice == "A":
            return payoff_matrix["cooperate_cooperate"][0]
        elif player_choice == "A" and partner_choice == "B":
            return payoff_matrix["cooperate_defect"][0]
        elif player_choice == "B" and partner_choice == "A":
            return payoff_matrix["defect_cooperate"][0]
        else:  # Both B
            return payoff_matrix["defect_defect"][0]

    def parse_choice(self, response_text: str, max_retries: int = 3) -> tuple[str, str]:
        """
        Parse the model's choice from response text with retries
        Returns tuple of (choice, reasoning)
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Look for explicit choice indicators
                choice_patterns = [
                    r"(?i)(?:choice|choose|select|opt for|decision):?\s*(?:option\s*)?([AB])",
                    r"(?i)I\s+(?:choose|select|pick|opt for)\s+(?:option\s*)?([AB])",
                    r"(?i)(?:option|choice)\s*([AB])\s+is\s+(?:better|preferable)",
                    r"(?i)Let's\s+(?:go with|choose)\s+(?:option\s*)?([AB])",
                    r"(?i)I\s+will\s+(?:go with|choose)\s+(?:option\s*)?([AB])",
                    r"(?i)(?:option|choice)\s*([AB])"
                ]
                
                for pattern in choice_patterns:
                    match = re.search(pattern, response_text)
                    if match:
                        choice = match.group(1).upper()
                        # Extract reasoning - everything after the choice
                        reasoning_start = match.end()
                        reasoning = response_text[reasoning_start:].strip()
                        if not reasoning:
                            # If no reasoning after choice, use everything before
                            reasoning = response_text[:match.start()].strip()
                        return choice, reasoning
                
                # If no choice found, retry
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Choice parsing failed. Retrying ({retry_count}/{max_retries})...")
                    continue
                else:
                    raise ValueError("Could not parse choice after maximum retries")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Error parsing choice: {str(e)}. Retrying ({retry_count}/{max_retries})...")
                    continue
                else:
                    raise ValueError(f"Failed to parse choice after {max_retries} attempts: {str(e)}")
        
        raise ValueError("Could not parse choice from response")

    def _generate_prompt(self, round_num: int, memory: str = "", condition: str = "neutral") -> str:
        """Generate prompt for the current round"""
        # Get condition-specific template
        template = self._get_prompt_template()
        
        # Add memory if available
        if memory:
            prompt_parts = template.split('\n\n', 1)
            if len(prompt_parts) > 1:
                template = f"{prompt_parts[0]}\n{memory}\n{prompt_parts[1]}"
            else:
                template = f"{template}\n{memory}"
        
        return template

def main():
    """Main function to run experiments"""
    runner = ExperimentRunner()
    experiment_id = runner.run_experiment()
    
    # Get and print results
    results = runner.experiment_manager.get_experiment_results(experiment_id)
    print(f"Experiment {experiment_id} completed. Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 