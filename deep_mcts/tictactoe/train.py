import datetime
from pathlib import Path

from deep_mcts.tictactoe.convolutionalnet import ConvolutionalTicTacToeNet
from deep_mcts.tictactoe.game import TicTacToeManager, TicTacToeState
from deep_mcts.train import train, TrainingConfiguration

if __name__ == "__main__":
    manager = TicTacToeManager()
    anet = ConvolutionalTicTacToeNet(manager)
    save_dir = (
        Path(__file__).resolve().parent
        / "saves"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    save_dir.mkdir()
    train(
        anet,
        TrainingConfiguration[TicTacToeState](
            num_games=5000,
            num_simulations=25,
            save_interval=10_000,
            evaluation_interval=10_000,
            save_dir=str(save_dir),
            sample_move_cutoff=3,
            dirichlet_alpha=2.5,
        ),
    )
