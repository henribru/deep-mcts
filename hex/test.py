from __future__ import annotations

import train
from hex import HexManager
from hex.convolutional import ConvolutionalHexANET


def hex_simulator(
    grid_size: int, num_actual_games: int, num_search_games: int, save_interval: int
) -> None:
    hex = HexManager(grid_size)
    anet = ConvolutionalHexANET(grid_size)
    for _ in train.train(
        anet, hex, num_actual_games, num_search_games, save_interval
    ):
        continue


if __name__ == "__main__":
    num_actual_games = 10
    hex_simulator(4, num_actual_games, num_search_games=100, save_interval=100)
    # state_manager = HexManager(4)
    # print(topp([ConvolutionalHexANET(4).sampling_policy, ConvolutionalHexANET(4).sampling_policy], 100, state_manager))
    # agents = [ConvolutionalHexANET.from_path(f"anet-{i}.pth", 4).sampling_policy for i in range(0, 110, 10)]
    # import pprint;pprint.pprint(topp(agents, 25, state_manager))
