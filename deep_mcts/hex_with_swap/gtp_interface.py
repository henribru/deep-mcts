import re
import string
import sys
from typing import Callable, Dict, List, Optional

from deep_mcts.gtp_interface import GTPInterface
from deep_mcts.hex_with_swap.convolutionalnet import ConvolutionalHexWithSwapNet
from deep_mcts.hex_with_swap.game import (
    Action,
    HexWithSwapManager,
    HexState,
)
from deep_mcts.mcts import MCTS


class HexWithSwapGTPInterface(GTPInterface[HexState]):
    commands: Dict[str, Callable[[List[str]], Optional[str]]]
    board_size: int
    game_manager: HexWithSwapManager
    mcts: MCTS[HexState]

    def __init__(self, board_size: int = 5) -> None:
        super().__init__(board_size)
        self.commands["hexgui-analyze_commands"] = self.analyze_commands

    def analyze_commands(self, args: List[str]) -> None:
        pass

    @staticmethod
    def parse_move(move: str, grid_size: int) -> Action:
        move = move.lower()
        if move == "swap":
            return grid_size ** 2
        match = re.match(r"([a-z])(\d{1,2})", move)
        if match is None:
            raise ValueError("invalid move")
        x, y = match.groups()
        x = string.ascii_lowercase.find(x)
        y = int(y) - 1
        if x >= grid_size or not 0 <= y < grid_size:
            raise ValueError("invalid move")
        return x + y * grid_size

    @staticmethod
    def format_move(move: Action, grid_size: int) -> str:
        if move == grid_size ** 2:
            return "swap"
        x, y = move % grid_size, move // grid_size
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalHexWithSwapNet:
        if board_size == int(sys.argv[1]):
            return ConvolutionalHexWithSwapNet.from_path(sys.argv[2], board_size)
        return ConvolutionalHexWithSwapNet(board_size)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        gtp_interface = HexWithSwapGTPInterface(board_size=int(sys.argv[1]))
    else:
        gtp_interface = HexWithSwapGTPInterface()
    gtp_interface.start()
