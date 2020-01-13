from pathlib import Path

from deep_mcts.evaluate_simple_rollouts import evaluate_simple_rollouts
from deep_mcts.hex_with_swap.game import HexWithSwapManager
from deep_mcts.hex_with_swap.convolutionalnet import ConvolutionalHexWithSwapNet


save_dir = Path(__file__).resolve().parent / "saves"
manager = HexWithSwapManager(grid_size=6)
evaluate_simple_rollouts(save_dir, ConvolutionalHexWithSwapNet, manager)
