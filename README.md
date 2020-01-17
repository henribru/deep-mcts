# Deep MCTS

This repository contains the code and raw data for my master's thesis *Deep reinforcement learning using Monte-Carlo tree search for Hex and Othello*. Due to storage restrictions the trained models from experiment 1 are not included.

## Raw data

### Experiment 1
<!-- The models can be found in subdirectories of `deep_mcst/<game>/saves` as files on the form `anet-<n>.tar`, where *n* denotes the number of iterations it has been trained for. They are stored as pickled dictionaries of parameters.  They can be loaded using the `from_path_full` method of `GameNet` subclasses. Many of the training parameters are also included in a `parameters.json` file in each subdirectory.-->
The evaluations are found as CSV files in `deep_mcts/<game>/training`. The format is specified by a header.

### Experiment 2
The evaluations are found as JSON files in `deep_mcts/<game>/simple_rollouts`, one for each model. The JSON describes a 6x6x3x2 array with the dimensions corresponding to:

1. Each rollout ratio
2. Each rollout ratio it was compared to
3. Wins, draws and losses
4. As the first player, as the second player

### Experiment 3
The evaluations are found as JSON files in `deep_mcts/<game>/complex_rollouts`, one for each model. The JSON describes a 3x2 array with the dimensions corresponding to:

1. Wins, draws and losses
2. As the first player, as the second player

## Running

### Setup
Using [Poetry](https://python-poetry.org/), simply run `poetry install`. If not, manually install the dependencies specified in `pyproject.toml`.

### Experiment 1
If running on a machine with only one GPU, set both `train_device` and `self_play_device` in `TrainingConfiguration` in `deep_mcts/train.py` to `cuda:0`.

1. `python -m deep_mcts.<game>.train`
2. `python -m deep_mcts.<game>.evaluate_training`

### Experiment 2
1. `python -m deep_mcts.<game>.evaluate_simple_rollouts`

### Experiment 3
1. `python -m deep_mcts.<game>.evaluate_complex_rollouts`
