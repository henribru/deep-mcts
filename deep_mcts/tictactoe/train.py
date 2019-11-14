from pathlib import Path
import datetime

from deep_mcts import train
from deep_mcts.tictactoe.convolutionalnet import ConvolutionalTicTacToeNet
from deep_mcts.tictactoe.game import TicTacToeManager


if __name__ == "__main__":
    manager = TicTacToeManager()
    anet = ConvolutionalTicTacToeNet(manager)
    save_dir = (
        Path(__file__).resolve().parent / "saves" / datetime.datetime.now().isoformat()
    )
    save_dir.mkdir()
    train.train(
        anet,
        num_games=5000,
        num_simulations=25,
        save_interval=10_000,
        evaluation_interval=10_000,
        save_dir=str(save_dir),
        sample_move_cutoff=3,
    )
