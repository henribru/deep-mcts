import os.path
import re
import string
from typing import Callable, Dict, List, Optional, Mapping

from deep_mcts.hex_with_swap.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex_with_swap.game import (
    HexAction,
    HexMove,
    HexSwap,
    HexWithSwapManager,
    HexState,
    hex_probabilities_grid,
)
from deep_mcts.mcts import MCTS
from deep_mcts.gtp_interface import GTPInterface


class HexWithSwapGTPInterface(GTPInterface[HexState, HexAction]):
    commands: Dict[str, Callable[[List[str]], Optional[str]]]
    board_size: int
    game_manager: HexWithSwapManager
    mcts: MCTS[HexState, HexAction]

    def __init__(self) -> None:
        super().__init__(board_size=5)
        self.commands["hexgui-analyze_commands"] = self.analyze_commands

    def analyze_commands(self, args: List[str]) -> None:
        pass

    @staticmethod
    def parse_move(move: str, grid_size: int) -> HexAction:
        move = move.lower()
        if move == "swap":
            return HexSwap()
        match = re.match(r"([a-z])(\d{1,2})", move)
        if match is None:
            raise ValueError("invalid move")
        x, y = match.groups()
        x = string.ascii_lowercase.find(x)
        y = int(y) - 1
        if x >= grid_size or not 0 <= y < grid_size:
            raise ValueError("invalid move")
        return HexMove((x, y))

    @staticmethod
    def format_move(move: HexAction) -> str:
        if isinstance(move, HexSwap):
            return "swap"
        x, y = move.coordinate
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_manager(board_size: int) -> HexWithSwapManager:
        return HexWithSwapManager(board_size)

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalHexNet:
        return ConvolutionalHexNet(board_size)

    @staticmethod
    def probabilities_grid(
        action_probabilities: Mapping[HexAction, float], board_size: int
    ) -> str:
        return hex_probabilities_grid(action_probabilities, board_size)


if __name__ == "__main__":
    gtp_interface = HexWithSwapGTPInterface()
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
