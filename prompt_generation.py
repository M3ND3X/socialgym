from typing import Dict, Any
import json
from game_history import GameHistory

class PromptGenerator:
    def __init__(self):
        self.payoff_templates = {
            "standard": {
                "cooperate_cooperate": (3, 3),
                "cooperate_defect": (0, 5),
                "defect_cooperate": (5, 0),
                "defect_defect": (1, 1)
            }
        }

    def generate_system_prompt(self, config: Dict[str, Any]) -> str:
        """Generate system prompt based on experiment configuration"""
        return ("You are participating in a competitive decision-making game. "
                "Your primary goal is to maximize your own points. "
                "While cooperation can be beneficial, don't hesitate to choose the option "
                "that gives you the highest personal gain. "
                "For each task, you will be given two options (A or B). "
                "Consider the risk/reward ratio carefully - sometimes taking risks can lead to better outcomes. "
                "Please provide your reasoning first, then state your choice clearly as either 'Option A' or 'Option B'.")

    def generate_user_prompt(
        self, 
        config: Dict[str, Any], 
        round_number: int,
        game_history: GameHistory
    ) -> str:
        """Generate user prompt with appropriate history"""
        condition = self._get_condition(config, round_number)
        
        if config["experiment_type"] == "memory_iterative":
            if condition["type"] == "no_memory":
                return condition["prompt_template"]
            
            elif condition["type"] == "short_term_memory":
                if not game_history.rounds:
                    return condition["prompt_template"].format(
                        current_round=round_number,
                        previous_choice="This is your first decision"
                    )
                last_round = game_history.rounds[-1]
                return condition["prompt_template"].format(
                    current_round=round_number,
                    previous_choice=f"In the previous round, you chose Option {last_round.player_choice.value} "
                    f"and your partner chose Option {last_round.partner_choice.value}"
                )
            
            elif condition["type"] == "long_term_memory":
                history = game_history.get_formatted_history()
                return condition["prompt_template"].format(
                    current_round=round_number,
                    n=len(history["player_history"]),
                    player_history=", ".join(f"Option {c}" for c in history["player_history"]),
                    partner_history=", ".join(f"Option {c}" for c in history["partner_history"])
                )
        
        # For other experiment types, just return the template
        return condition["prompt_template"]

    def _get_condition(self, config: Dict[str, Any], round_number: int) -> Dict[str, Any]:
        """Get the appropriate condition based on experiment type and round number"""
        conditions = config.get("conditions", {})
        experiment_type = config.get("experiment_type", "")
        
        # For memory_iterative experiments, select condition based on type
        if experiment_type == "memory_iterative":
            memory_type = config.get("memory_type", "no_memory")
            if memory_type in conditions:
                return conditions[memory_type]
            return conditions["no_memory"]
        
        # For other experiments, use specified condition type if available
        condition_type = config.get("condition_type")
        if condition_type and condition_type in conditions:
            return conditions[condition_type]
        
        # Fall back to first condition as default
        return next(iter(conditions.values())) 