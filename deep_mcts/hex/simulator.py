from deep_mcts import train
from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex.game import HexManager
import pandas as pd


def hex_simulator(
    grid_size: int,
    num_games: int,
    num_simulations: int,
    save_interval: int,
    evaluation_interval: int,
) -> None:
    manager = HexManager(grid_size)
    anet = ConvolutionalHexNet(grid_size)
    evaluations = pd.DataFrame.from_dict(
        {
            i: (random_evaluation, previous_evaluation)
            for i, random_evaluation, previous_evaluation in train.train(
                anet,
                manager,
                num_games,
                num_simulations,
                save_interval,
                evaluation_interval,
            )
        },
        orient="index",
        columns=["against_random", "against_previous"],
    )
    evaluations.to_csv("evaluations2.csv")


if __name__ == "__main__":
    num_actual_games = 1000
    hex_simulator(
        5,
        num_actual_games,
        num_simulations=100,
        save_interval=100,
        evaluation_interval=5,
    )
    # state_manager = HexManager(4)
    # print(topp([ConvolutionalHexNet(4).sampling_policy, ConvolutionalHexNet(4).sampling_policy], 100, state_manager))
    # agents = [ConvolutionalHexNet.from_path(f"anet-{i}.pth", 4).sampling_policy for i in range(0, 110, 10)]
    # import pprint;pprint.pprint(topp(agents, 25, state_manager))