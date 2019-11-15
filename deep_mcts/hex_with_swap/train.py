from pathlib import Path
import datetime

from deep_mcts import train
from deep_mcts.hex_with_swap.convolutionalnet import ConvolutionalHexWithSwapNet
from deep_mcts.hex_with_swap.game import HexWithSwapManager


if __name__ == "__main__":
    grid_size = 5
    manager = HexWithSwapManager(grid_size)
    anet = ConvolutionalHexWithSwapNet(grid_size, manager)
    save_dir = (
        Path(__file__).resolve().parent
        / "saves"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    save_dir.mkdir()
    train.train(
        anet,
        num_games=5000,
        num_simulations=25,
        save_interval=10_000,
        evaluation_interval=10_000,
        save_dir=str(save_dir),
        sample_move_cutoff=10,
    )
