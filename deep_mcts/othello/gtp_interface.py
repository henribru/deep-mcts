import dataclasses
import os.path
import re
import string
from typing import Mapping

from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.othello.game import (
    OthelloAction,
    OthelloMove,
    OthelloPass,
    OthelloManager,
    OthelloState,
)
from deep_mcts.gtp_interface import GTPInterface
from deep_mcts.othello.game import othello_probabilities_grid


class OthelloGTPInterface(GTPInterface[OthelloState, OthelloAction]):
    def __init__(self) -> None:
        super().__init__(board_size=6)

    @staticmethod
    def parse_move(move: str, board_size: int) -> OthelloAction:
        move = move.lower()
        if move == "pass":
            return OthelloPass()
        match = re.match(r"([a-z])(\d{1,2})", move)
        if match is None:
            raise ValueError("invalid move")
        x, y = match.groups()
        x = string.ascii_lowercase.find(x)
        y = int(y) - 1
        if x >= board_size or not 0 <= y < board_size:
            raise ValueError("invalid move")
        return OthelloMove((x, y))

    @staticmethod
    def format_move(move: OthelloAction) -> str:
        if isinstance(move, OthelloPass):
            return "pass"
        x, y = move.coordinate
        return f"{string.ascii_lowercase[x]}{y + 1}"

    @staticmethod
    def get_game_manager(board_size: int) -> OthelloManager:
        return OthelloManager(board_size)

    @staticmethod
    def get_game_net(board_size: int) -> ConvolutionalOthelloNet:
        if board_size == 6:
            return ConvolutionalOthelloNet.from_path(
                os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "saves",
                    "anet-130000.pth",
                ),
                board_size,
            )
        return ConvolutionalOthelloNet(board_size)

    @staticmethod
    def probabilities_grid(
        action_probabilities: Mapping[OthelloAction, float], board_size: int
    ) -> str:
        return othello_probabilities_grid(action_probabilities, board_size)


if __name__ == "__main__":
    gtp_interface = OthelloGTPInterface()
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
