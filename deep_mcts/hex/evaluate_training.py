from pathlib import Path

import torch

from deep_mcts.hex.game import HexManager
from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.evaluate_training import evaluate_training

evaluate_training(
    Path(__file__).resolve().parent / "saves",
    ConvolutionalHexNet,
    HexManager(grid_size=6),
    torch.device("cuda:0"),
)
