import os.path
import re
import string
from typing import Mapping, List

from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex.game import HexAction, HexManager, HexState
from deep_mcts.hex.game import hex_probabilities_grid
from deep_mcts.gtp_interface import GTPInterface


class HexGTPInterface(GTPInterface[HexState, HexAction]):
    def __init__(self) -> None:
        super().__init__(board_size=5)
        self.commands["hexgui-analyze_commands"] = self.analyze_commands

    def analyze_commands(self, args: List[str]) -> None:
        pass

    @staticmethod
    def parse_move(move: str, grid_size: int) -> HexAction:
        move = move.lower()
        match = re.match(r"([a-z])(\d{1,2})", move)
        if match is None:
            raise ValueError("invalid move")
        x, y = match.groups()
        x = string.ascii_lowercase.find(x)
        y = int(y) - 1
        if x >= grid_size or not 0 <= y < grid_size:
            raise ValueError("invalid move")
        return HexAction((x, y))

    @staticmethod
    def format_move(move: HexAction) -> str:
        x, y = move.coordinate
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_manager(board_size: int) -> HexManager:
        return HexManager(board_size)

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalHexNet:
        if board_size == 5:
            return ConvolutionalHexNet.from_path(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "saves",
                    "anet-364000.pth",
                ),
                board_size,
            )
        return ConvolutionalHexNet(board_size)

    @staticmethod
    def probabilities_grid(
        action_probabilities: Mapping[HexAction, float], board_size: int
    ) -> str:
        return hex_probabilities_grid(action_probabilities, board_size)


if __name__ == "__main__":
    gtp_interface = HexGTPInterface()
    while True:
        command = input("")
        if not command:
            continue
        try:
            result = gtp_interface.run_command(command)
        except ValueError as e:
            print(f"? {e}\n")
        else:
            if result is None:
                print("= \n")
            else:
                print(f"= {result}\n")
