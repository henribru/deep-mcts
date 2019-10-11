import re
import string
import sys
from typing import Tuple, List, Callable, Dict, NoReturn, Optional

import dataclasses

from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex.game import HexManager, HexAction
from deep_mcts.mcts import MCTS, Node


class GTPInterface:
    commands: Dict[str, Callable[["GTPInterface", List[str]], Optional[str]]]
    board_size: int

    def __init__(self):
        self.commands = {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "known_command": self.known_command,
            "list_commands": self.list_commands,
            "quit": self.quit,
            "boardsize": self.boardsize,
            "clear_board": self.clear_board,
            "play": self.play,
            "genmove": self.genmove,
            "showboard": self.showboard,
            # "set_time": self.set_time,
            # "winner": self.winner,
            "hexgui-analyze_commands": self.analyze_commands
        }
        self.board_size = 5
        self.game_manager = HexManager(self.board_size)
        self.mcts = MCTS(
            self.game_manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=ConvolutionalHexNet(self.board_size).evaluate_state,
        )

    def run_command(self, command) -> str:
        command, *args = command.split()
        if command in self.commands:
            return self.commands[command](args)
        else:
            raise ValueError("invalid command")

    def name(self, args: List[str]) -> str:
        return "Deep MCTS"

    def version(self, args: List[str]) -> str:
        return "0.0.1"

    def protocol_version(self, args: List[str]) -> str:
        return "2"

    def known_command(self, args: List[str]) -> str:
        if len(args) != 1:
            raise ValueError("known_command takes 1 argument")
        command = args[0]
        known = command in self.commands
        return str(known).lower()

    def list_commands(self, args: List[str]) -> str:
        return "\n".join(list(self.commands))

    def quit(self, args: List[str]) -> NoReturn:
        sys.exit(0)

    def boardsize(self, args: List[str]) -> None:
        if len(args) != 1:
            raise ValueError("boardsize takes 1 argument")
        board_size = int(args[0])
        if board_size < 1:
            raise ValueError("invalid board size")
        self.board_size = board_size

    def clear_board(self, args: List[str]) -> None:
        self.manager = HexManager(self.board_size)
        self.mcts = MCTS(
            self.manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=ConvolutionalHexNet(self.board_size).evaluate_state,
        )

    def play(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ValueError("play takes 2 arguments")
        player = parse_player(args[0])
        x, y = parse_move(args[1])
        if x >= self.board_size or not 0 < y < self.board_size:
            raise ValueError("invalid move")
        action = HexAction((x, y))
        legal_actions = self.game_manager.legal_actions(self.mcts.root.state)
        if action not in legal_actions:
            raise ValueError("illegal move")
        actual_player = self.mcts.root.state.player
        if actual_player != player:
            self.mcts.root = Node(dataclasses.replace(self.mcts.root.state, player=player))
        state = self.game_manager.generate_child_state(self.mcts.root.state, action)
        self.mcts.root = next(
            (
                child
                for child in self.mcts.root.children.values()
                if child.state == state
            ),
            Node(state),
        )

    def genmove(self, args: List[str]) -> str:
        if len(args) != 1:
            raise ValueError("play takes 1 argument")
        player = parse_player(args[0])
        actual_player = self.mcts.root.state.player
        if actual_player != player:
            self.mcts.root = Node(dataclasses.replace(self.mcts.root.state, player=player))
        action_probabilities = self.mcts.step()
        action = max(action_probabilities.keys(), key=lambda a: action_probabilities[a])
        self.mcts.root = self.mcts.root.children[action]
        return format_move(action.coordinate)

    def showboard(self, args: List[str]) -> str:
        return f"\n{self.mcts.root.state}"

    def analyze_commands(self, args: List[str]) -> None:
        pass


def parse_player(player: str) -> int:
    player = player.lower()
    if player == "w":
        player = "white"
    elif player == "b":
        player = "black"
    if player not in ["white", "black"]:
        raise ValueError("invalid player")
    # TODO?
    return 0 if player == "white" else 1


def parse_move(move: str) -> Tuple[int, int]:
    move = move.lower()
    match = re.match(r"([a-z])(\d{1,2})", move)
    if match is None:
        raise ValueError("invalid move")
    x, y = match.groups()
    # TODO?
    x = string.ascii_lowercase.find(x)
    y = int(y) - 1
    return x, y


def format_move(move: Tuple[int, int]) -> str:
    x, y = move
    return f"{string.ascii_lowercase[x]}{y + 1}"


if __name__ == '__main__':
    gtp_interface = GTPInterface()
    while True:
        command = input("> ")
        try:
            result = gtp_interface.run_command(command)
        except ValueError as e:
            print(f"? {e}")
        else:
            if result is not None:
                print(f"= {result}")
