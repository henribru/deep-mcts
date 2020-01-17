from pathlib import Path

import torch

from deep_mcts.othello.game import OthelloManager
from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.evaluate_complex_rollouts import evaluate_complex_rollouts

evaluate_complex_rollouts(
    Path(__file__).resolve().parent / "saves",
    ConvolutionalOthelloNet,
    OthelloManager(grid_size=6),
    torch.device("cuda:1"),
)
