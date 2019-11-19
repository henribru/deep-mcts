import re
import re
import string
import sys
from typing import List

from deep_mcts.gtp_interface import GTPInterface
from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex.game import Action, HexState


class HexGTPInterface(GTPInterface[HexState]):
    def __init__(self, board_size: int = 5) -> None:
        super().__init__(board_size)
        self.commands["hexgui-analyze_commands"] = self.analyze_commands

    def analyze_commands(self, args: List[str]) -> None:
        pass

    @staticmethod
    def parse_move(move: str, grid_size: int) -> Action:
        move = move.lower()
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
    def format_move(move: Action, board_size: int) -> str:
        x, y = move % board_size, move // board_size
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalHexNet:
        if board_size == int(sys.argv[1]):
            return ConvolutionalHexNet.from_path(sys.argv[2], board_size)
        return ConvolutionalHexNet(board_size)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        gtp_interface = HexGTPInterface(board_size=int(sys.argv[1]))
    else:
        gtp_interface = HexGTPInterface()
    gtp_interface.start()
