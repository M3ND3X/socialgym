# LLM Decision-Making Experiments

A framework for running and analyzing decision-making experiments with Large Language Models (LLMs). This project investigates how different LLMs behave in various social and economic decision-making scenarios.

## Project Structure

```
thesis/
├── config/                     # Configuration files
│   ├── model_config.json      # LLM model configurations
│   ├── risk_aversion.json     # Risk aversion experiment
│   ├── social_preferences.json # Social preferences experiment
│   ├── memory_iterative.json  # Memory-based iterative games
│   ├── framing_effects.json   # Framing effects experiment
│   └── ...                    # Other experiment configs
├── results/
│   └── test_runs/            # Experiment results
├── scripts/
│   └── data_analysis.py      # Analysis utilities
├── test_runner.py            # Main experiment runner
├── xtest.py                  # Results export utility
├── experiment_runner.py      # Core experiment logic
├── prompt_generation.py      # Prompt management
├── model_interface.py        # LLM API interface
├── game_history.py          # Game state tracking
└── experiment_management.py  # Experiment orchestration
```

## Experiment Types

1. **Risk Aversion**
   - Tests LLM decision-making under different risk levels
   - Conditions: low_risk, high_risk, uncertain

2. **Social Preferences**
   - Examines social dynamics in decision-making
   - Conditions: equal_status, high_status, low_status

3. **Memory Iterative**
   - Studies impact of memory on repeated decisions
   - Conditions: no_memory, short_term_memory, long_term_memory

4. **Framing Effects**
   - Investigates how prompt framing affects choices
   - Conditions: positive, negative, neutral

5. **Political Cultural**
   - Tests influence of cultural context
   - Conditions: individualistic, collectivistic

6. **Default Bias**
   - Examines impact of default options
   - Conditions: default_mutual, default_self, no_default

7. **Temporal Preferences**
   - Studies time-based decision-making
   - Conditions: immediate, delayed

8. **Emotional Priming**
   - Tests effect of emotional context
   - Conditions: neutral, empathy, competitive, collaborative

9. **Peer Influence**
   - Examines social influence on decisions
   - Conditions: no_influence, majority_a, majority_b, split_influence

## Usage

### Running Experiments

```bash
# Run a basic test
python test_runner.py --utest --config risk_aversion --r 7 --condition high_risk

# Run with memory
python test_runner.py --utest --config memory_iterative --r 20 --memory-type long_term_memory

# Export results to Excel
python xtest.py --ex 20250103_025847_memory_iterative
```

### Configuration

Each experiment type has a JSON configuration file with:
- Experiment parameters
- Condition definitions
- Prompt templates
- Payoff matrices

Example config:
```json
{
    "experiment_type": "risk_aversion",
    "num_iterations": 30,
    "condition_type": "high_risk",
    "conditions": {
        "low_risk": {
            "type": "low_risk",
            "prompt_template": "..."
        }
    },
    "payoff_matrix": {
        "cooperate_cooperate": [3, 3],
        "cooperate_defect": [0, 5],
        "defect_cooperate": [5, 0],
        "defect_defect": [1, 1]
    }
}
```

### Analysis

Results are exported to Excel with multiple sheets:
- Summary: Overall experiment statistics
- Choices: Detailed choice analysis
- Reasoning: Model reasoning analysis
- Model Performance: Configuration metrics
- Round Details: Complete interaction logs

## Dependencies

```
google-generativeai
pandas
numpy
python-dotenv
matplotlib
seaborn
scipy 
tqdm
```

## Installation

```bash
git clone <repository-url>
cd thesis
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

mendexofficial@gmail.com | +251982838683
