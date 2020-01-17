from pathlib import Path

import torch

from deep_mcts.evaluate_simple_rollouts import evaluate_simple_rollouts
from deep_mcts.hex.game import HexManager
from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet


save_dir = Path(__file__).resolve().parent / "saves"
manager = HexManager(grid_size=6)
evaluate_simple_rollouts(save_dir, ConvolutionalHexNet, manager, torch.device("cuda:0"))
