from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class Choice(Enum):
    A = "A"
    B = "B"

@dataclass
class Round:
    player_choice: Choice
    partner_choice: Choice
    player_score: int
    partner_score: int
    round_number: int

class GameHistory:
    def __init__(self):
        self.rounds = []
        
    @property
    def player_history(self) -> List[str]:
        """Get list of player's choices"""
        return [r.player_choice.value for r in self.rounds]
        
    @property
    def partner_history(self) -> List[str]:
        """Get list of partner's choices"""
        return [r.partner_choice.value for r in self.rounds]
    
    def add_round(self, player_choice: str, partner_choice: str, round_number: int) -> None:
        """Add a round to the history"""
        # Convert string choices to Choice enum
        p_choice = Choice.A if "a" in player_choice.lower() else Choice.B
        o_choice = Choice.A if "a" in partner_choice.lower() else Choice.B
        
        # Calculate scores based on choices
        p_score, o_score = self._calculate_scores(p_choice, o_choice)
        
        # Add round to history
        self.rounds.append(Round(
            player_choice=p_choice,
            partner_choice=o_choice,
            player_score=p_score,
            partner_score=o_score,
            round_number=round_number
        ))
    
    def _calculate_scores(self, player_choice: Choice, partner_choice: Choice) -> Tuple[int, int]:
        """Calculate scores for a round based on choices"""
        payoff_matrix = {
            (Choice.A, Choice.A): (3, 3),
            (Choice.A, Choice.B): (0, 5),
            (Choice.B, Choice.A): (5, 0),
            (Choice.B, Choice.B): (1, 1)
        }
        return payoff_matrix[(player_choice, partner_choice)]
    
    def get_outcome_type(self, player_choice: str, partner_choice: str) -> str:
        """Calculate outcome type based on choices
        
        Returns one of:
        - "MutualCooperation"
        - "MutualDefection"
        - "PlayerExploited"
        - "PartnerExploited"
        """
        if player_choice == 'A' and partner_choice == 'A':
            return "MutualCooperation"
        elif player_choice == 'B' and partner_choice == 'B':
            return "MutualDefection"
        elif player_choice == 'A' and partner_choice == 'B':
            return "PlayerExploited"
        elif player_choice == 'B' and partner_choice == 'A':
            return "PartnerExploited"
        return None  # This shouldn't happen, but included for safety
    
    def get_cooperation_rate(self, player_type: str = "player") -> float:
        """Calculate the cooperation rate for player or partner
        
        Args:
            player_type: "player" or "partner"
            
        Returns:
            The proportion of 'A' choices (cooperation) in history
        """
        if not self.rounds:
            return 0.0
            
        if player_type == "player":
            choices = [r.player_choice.value for r in self.rounds]
        else:  # partner
            choices = [r.partner_choice.value for r in self.rounds]
            
        return choices.count('A') / len(choices)
    
    def get_cumulative_score(self, player_type: str = "player") -> int:
        """Calculate the cumulative score for player or partner
        
        Args:
            player_type: "player" or "partner"
            
        Returns:
            The sum of scores in history
        """
        if not self.rounds:
            return 0
            
        if player_type == "player":
            return sum(r.player_score for r in self.rounds)
        else:  # partner
            return sum(r.partner_score for r in self.rounds)
    
    def get_last_n_rounds(self, n: int) -> List[Round]:
        """Get the last n rounds of history"""
        return self.rounds[-n:] if n > 0 else []
    
    def get_formatted_history(self, n: int = None) -> Dict[str, Any]:
        """Get formatted history for prompt generation"""
        rounds = self.rounds if n is None else self.get_last_n_rounds(n)
        
        return {
            "player_history": [r.player_choice.value for r in rounds],
            "partner_history": [r.partner_choice.value for r in rounds],
            "player_scores": [r.player_score for r in rounds],
            "partner_scores": [r.partner_score for r in rounds],
            "total_rounds": len(self.rounds)
        } 