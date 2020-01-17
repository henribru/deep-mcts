from pathlib import Path

import torch

from deep_mcts.othello.game import OthelloManager
from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.evaluate_training import evaluate_training

evaluate_training(
    Path(__file__).resolve().parent / "saves",
    ConvolutionalOthelloNet,
    OthelloManager(grid_size=6),
    torch.device("cuda:1"),
)
