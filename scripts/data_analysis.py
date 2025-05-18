import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

class ExperimentAnalyzer:
    def __init__(self, experiment_results: Dict[str, Any]):
        """Initialize analyzer with experiment results"""
        self.experiment = experiment_results['experiment']
        self.rounds = experiment_results['rounds']
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert experiment data to pandas DataFrame"""
        # Convert rounds to DataFrame
        df = pd.DataFrame(self.rounds)
        
        # Add experiment metadata
        for key, value in zip(self.experiment.keys(), self.experiment):
            df[key] = value
            
        return df
        
    def calculate_cooperation_rate(self) -> float:
        """Calculate overall cooperation rate"""
        decisions = self.df['raw_model_response'].apply(
            lambda x: 1 if 'cooperate' in x.lower() else 0
        )
        return decisions.mean()
        
    def analyze_by_condition(self, condition: str) -> pd.DataFrame:
        """Analyze cooperation rates by experimental condition"""
        grouped = self.df.groupby(condition)
        
        cooperation_rates = grouped['raw_model_response'].apply(
            lambda x: sum('cooperate' in resp.lower() for resp in x) / len(x)
        )
        
        return pd.DataFrame({
            'cooperation_rate': cooperation_rates,
            'sample_size': grouped.size()
        })
        
    def run_chi_square_test(self, condition: str) -> Dict[str, float]:
        """Run chi-square test for independence"""
        contingency = pd.crosstab(
            self.df[condition],
            self.df['raw_model_response'].apply(
                lambda x: 'cooperate' if 'cooperate' in x.lower() else 'defect'
            )
        )
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof
        }
        
    def plot_cooperation_rates(self, condition: str, title: str = None) -> None:
        """Plot cooperation rates by condition"""
        rates = self.analyze_by_condition(condition)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=rates.index, y='cooperation_rate', data=rates)
        plt.title(title or f'Cooperation Rates by {condition}')
        plt.ylabel('Cooperation Rate')
        plt.xlabel(condition)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/figures/cooperation_rates_{condition}.png') 