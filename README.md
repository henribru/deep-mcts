# Deep MCTS

This repository contains the code and raw data for my master's thesis *Deep reinforcement learning using Monte-Carlo tree search for Hex and Othello*. 

## Raw data and models

### Experiment 1
Due to storage restrictions the trained models from experiment 1 are not in the repository, but can instead be found in [OneDrive](https://1drv.ms/u/s!AleiVuil950KhVQMM3D6DejLhtK3?e=5F6DW4). Note that only the final models are included, not checkpoints.
The models are located in subdirectories of `deep_mcst/<game>/saves` as files on the form `anet-<n>.tar`, where *n* denotes the number of iterations it has been trained for. They are stored as pickled dictionaries of parameters.  They can be loaded using the `from_path_full` method of `GameNet` subclasses. Many of the training parameters are also included in a `parameters.json` file in each subdirectory.
The post-training evaluations are found as CSV files in `deep_mcts/<game>/training` in the repository. The format is specified by the header.

### Experiment 2
The evaluations are found as JSON files in `deep_mcts/<game>/simple_rollouts`, one for each model. The JSON describes a 6x6x3x2 array with the dimensions corresponding to:

1. Each rollout probability
2. Each rollout probability it was compared to
3. Wins, draws and losses for the rollout probability in the first dimension
4. As the first player, as the second player

### Experiment 3
The evaluations are found as JSON files in the two subdirectories of `deep_mcts/<game>/complex_rollouts`. There is one subdirectory for evaluations with a state evaluator and one without, and one file for each model in each folder. The JSON describes an object, where the "results" key corresponds to a 3x2 array with the dimensions corresponding to:

1. Wins, draws and losses for the policy network rollouts
2. As the first player, as the second player

Additionally there are "complex_simulations" and "simple_simulations" keys, corresponding to the number of simulations with and without expansion in each move of each game for policy network rollouts and random rollouts respectively.

## Running

### Setup
If using [Poetry](https://python-poetry.org/), run `poetry install`. If not, run `pip install -r requirements.txt`.

### Experiment 1
If running on a machine with only one GPU, set both `train_device` and `self_play_device` in `TrainingConfiguration` in `deep_mcts/train.py` to `cuda:0`.

1. `python -m deep_mcts.<game>.train`
2. `python -m deep_mcts.<game>.evaluate_training`

### Experiment 2
1. `python -m deep_mcts.<game>.evaluate_simple_rollouts`

### Experiment 3
1. `python -m deep_mcts.<game>.evaluate_complex_rollouts`
