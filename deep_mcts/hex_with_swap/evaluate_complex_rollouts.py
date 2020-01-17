from pathlib import Path

import torch

from deep_mcts.hex_with_swap.game import HexWithSwapManager
from deep_mcts.hex_with_swap.convolutionalnet import ConvolutionalHexWithSwapNet
from deep_mcts.evaluate_complex_rollouts import evaluate_complex_rollouts

evaluate_complex_rollouts(
    Path(__file__).resolve().parent / "saves",
    ConvolutionalHexWithSwapNet,
    HexWithSwapManager(grid_size=6),
    torch.device("cuda:1"),
)
