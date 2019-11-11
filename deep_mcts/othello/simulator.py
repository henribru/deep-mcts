import os.path

import pandas as pd

from deep_mcts import train
from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.othello.game import OthelloManager


def othello_simulator(
    grid_size: int,
    num_games: int,
    num_simulations: int,
    save_interval: int,
    evaluation_interval: int,
) -> None:
    manager = OthelloManager(grid_size)
    anet = ConvolutionalOthelloNet(grid_size, manager)
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "saves")
    evaluations = pd.DataFrame.from_dict(
        {
            i: (random_evaluation, previous_evaluation)
            for i, random_evaluation, previous_evaluation in train.train(
                anet,
                num_games,
                num_simulations,
                save_interval,
                evaluation_interval,
                save_dir,
            )
        },
        orient="index",
        columns=["against_random", "against_previous"],
    )
    evaluations.to_csv("evaluations2.csv")


if __name__ == "__main__":
    num_actual_games = 1000
    othello_simulator(
        6,
        num_actual_games,
        num_simulations=25,
        save_interval=10000,
        evaluation_interval=100,
    )
