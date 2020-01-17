from pathlib import Path

import torch

from deep_mcts.evaluate_simple_rollouts import evaluate_simple_rollouts
from deep_mcts.othello.game import OthelloManager
from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet


save_dir = Path(__file__).resolve().parent / "saves"
manager = OthelloManager(grid_size=6)
evaluate_simple_rollouts(
    save_dir, ConvolutionalOthelloNet, manager, torch.device("cuda:1")
)
