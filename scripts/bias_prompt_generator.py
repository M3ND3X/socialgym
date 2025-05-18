from typing import Dict, Any
import json

class BiasPromptGenerator:
    def __init__(self, experiment_type: str):
        """Initialize prompt generator for specific bias experiment"""
        self.experiment_type = experiment_type
        self.payoff_matrix = {
            "cooperate_cooperate": (3, 3),
            "cooperate_defect": (0, 5),
            "defect_cooperate": (5, 0),
            "defect_defect": (1, 1)
        }
        
    def generate_system_prompt(self, config: Dict[str, Any]) -> str:
        """Generate system prompt based on experiment type"""
        if self.experiment_type == "framing_effects":
            return self._generate_framing_system_prompt(config)
        elif self.experiment_type == "social_preferences":
            return self._generate_social_system_prompt(config)
        # Add other experiment types as needed
        
    def generate_user_prompt(self, config: Dict[str, Any], 
                           condition: str, round_number: int = 1,
                           history: Dict[str, Any] = None) -> str:
        """Generate user prompt based on experiment type and condition"""
        if self.experiment_type == "framing_effects":
            return self._generate_framing_user_prompt(config, condition)
        elif self.experiment_type == "social_preferences":
            return self._generate_social_user_prompt(config, condition, history)
        # Add other experiment types as needed
        
    def _generate_framing_system_prompt(self, config: Dict[str, Any]) -> str:
        """Generate system prompt for framing effects experiment"""
        return """You are participating in a decision-making experiment. 
Your choices will be analyzed to understand how different descriptions 
of the same situation might influence decision-making."""
        
    def _generate_social_system_prompt(self, config: Dict[str, Any]) -> str:
        """Generate system prompt for social preferences experiment"""
        return """You are participating in a series of interactions where
your decisions affect both your own outcomes and those of others. Consider
how your choices impact all participants."""
        
    def _generate_framing_user_prompt(self, config: Dict[str, Any], 
                                    condition: str) -> str:
        """Generate user prompt for framing effects experiment"""
        return config["conditions"][condition]["prompt_template"]
        
    def _generate_social_user_prompt(self, config: Dict[str, Any],
                                   condition: str,
                                   history: Dict[str, Any] = None) -> str:
        """Generate user prompt for social preferences experiment"""
        base_prompt = config["conditions"][condition]["prompt_template"]
        
        if history and condition == "reciprocity":
            # Add history information for reciprocity condition
            history_str = self._format_history(history)
            base_prompt = f"{history_str}\n\n{base_prompt}"
            
        return base_prompt
        
    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format interaction history for prompts"""
        if not history:
            return ""
            
        rounds = len(history.get("choices", []))
        if rounds == 0:
            return ""
            
        return f"In the previous {rounds} rounds, your partner has chosen to cooperate {history['cooperation_count']} times." 