import re
import string
import sys

from deep_mcts.game import Action
from deep_mcts.gtp_interface import GTPInterface
from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.othello.game import OthelloState


class OthelloGTPInterface(GTPInterface[OthelloState]):
    def __init__(self, board_size: int = 6) -> None:
        super().__init__(board_size)

    @staticmethod
    def parse_move(move: str, board_size: int) -> Action:
        move = move.lower()
        if move == "pass":
            return board_size ** 2
        match = re.match(r"([a-z])(\d{1,2})", move)
        if match is None:
            raise ValueError("invalid move")
        x, y = match.groups()
        x = string.ascii_lowercase.find(x)
        y = int(y) - 1
        if x >= board_size or not 0 <= y < board_size:
            raise ValueError("invalid move")
        return x + y * board_size

    @staticmethod
    def format_move(move: Action, board_size: int) -> str:
        if move == board_size ** 2:
            return "pass"
        x, y = move % board_size, move // board_size
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalOthelloNet:
        if len(sys.argv) == 3 and board_size == int(sys.argv[1]):
            path = sys.argv[2]
            if path.endswith(".pth"):
                return ConvolutionalOthelloNet.from_path(path, board_size)
            else:
                return ConvolutionalOthelloNet.from_path_full(path)
        return ConvolutionalOthelloNet(board_size)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        gtp_interface = OthelloGTPInterface(board_size=int(sys.argv[1]))
    else:
        gtp_interface = OthelloGTPInterface()
    gtp_interface.start()
