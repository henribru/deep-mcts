import os.path

import pandas as pd

from deep_mcts import train
from deep_mcts.tictactoe.convolutionalnet import ConvolutionalTicTacToeNet
from deep_mcts.tictactoe.game import TicTacToeManager


def tic_tac_toe_simulator(
    num_actual_games: int,
    num_search_games: int,
    save_interval: int,
    evaluation_interval: int,
) -> None:
    manager = TicTacToeManager()
    anet = ConvolutionalTicTacToeNet()
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "saves")
    evaluations = pd.DataFrame.from_dict(
        {
            i: (random_evaluation, previous_evaluation)
            for i, random_evaluation, previous_evaluation in train.train(
                anet,
                manager,
                num_actual_games,
                num_search_games,
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
    num_actual_games = 100_000
    tic_tac_toe_simulator(
        num_actual_games,
        num_search_games=100,
        save_interval=100,
        evaluation_interval=100,
    )
    # state_manager = HexManager(4)
    # print(topp([ConvolutionalHexNet(4).sampling_policy, ConvolutionalHexNet(4).sampling_policy], 100, state_manager))
    # agents = [ConvolutionalHexNet.from_path(f"anet-{i}.pth", 4).sampling_policy for i in range(0, 110, 10)]
    # import pprint;pprint.pprint(topp(agents, 25, state_manager))
