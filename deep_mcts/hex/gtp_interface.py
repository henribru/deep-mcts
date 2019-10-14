import re
import string
import sys
from typing import Tuple, List, Callable, Dict, NoReturn, Optional

import dataclasses
import pexpect

from deep_mcts.hex.convolutionalnet import ConvolutionalHexNet
from deep_mcts.hex.game import HexManager, HexAction, HexState
from deep_mcts.mcts import MCTS, Node
from deep_mcts.tournament import Agent


class GTPInterface:
    commands: Dict[str, Callable[[List[str]], Optional[str]]]
    board_size: int
    game_manager: HexManager
    mcts: MCTS[HexState, HexAction]

    def __init__(self) -> None:
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
            "hexgui-analyze_commands": self.analyze_commands,
        }
        self.board_size = 5
        self.game_manager = HexManager(self.board_size)
        self.mcts = MCTS(
            self.game_manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=ConvolutionalHexNet.from_path(
                "saves/anet-50400.pth", self.board_size
            ).evaluate_state,
        )

    def run_command(self, command: str) -> Optional[str]:
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
        return "\n" + "\n".join(list(self.commands))

    def quit(self, args: List[str]) -> NoReturn:
        sys.exit(0)

    def boardsize(self, args: List[str]) -> None:
        # allow 2 arguments because HexGui passes the board size twice
        if len(args) not in [1, 2]:
            raise ValueError("boardsize takes 1 argument")
        try:
            board_size = int(args[0])
        except ValueError:
            raise ValueError("invalid board size")
        if board_size < 1:
            raise ValueError("invalid board size")
        self.board_size = board_size
        self.clear_board([])

    def clear_board(self, args: List[str]) -> None:
        self.game_manager = HexManager(self.board_size)
        if self.board_size == 5:
            game_net = ConvolutionalHexNet.from_path(
                "saves/anet-50400.pth", self.board_size
            )
        else:
            game_net = ConvolutionalHexNet(self.board_size)
        self.mcts = MCTS(
            self.game_manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=game_net.evaluate_state,
        )

    def play(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ValueError("play takes 2 arguments")
        player = parse_player(args[0])
        x, y = parse_move(args[1], self.board_size)
        action = HexAction((x, y))
        legal_actions = self.game_manager.legal_actions(self.mcts.root.state)
        if action not in legal_actions:
            raise ValueError("illegal move")
        actual_player = self.mcts.root.state.player
        if actual_player != player:
            self.mcts.root = Node(
                dataclasses.replace(self.mcts.root.state, player=player)
            )
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
            self.mcts.root = Node(
                dataclasses.replace(self.mcts.root.state, player=player)
            )
        action_probabilities = self.mcts.step()
        action = max(action_probabilities.keys(), key=lambda a: action_probabilities[a])
        self.mcts.root = self.mcts.root.children[action]
        return format_move(action.coordinate, self.board_size)

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
    return 0 if player == "black" else 1


def format_player(player: int) -> str:
    return "black" if player == 0 else "white"


def parse_move(move: str, grid_size: int) -> Tuple[int, int]:
    move = move.lower()
    match = re.match(r"([a-z])(\d{1,2})", move)
    if match is None:
        raise ValueError("invalid move")
    y, x = match.groups()
    y = grid_size - 1 - string.ascii_lowercase.find(y)
    x = int(x) - 1
    if y >= grid_size or not 0 <= x < grid_size:
        raise ValueError("invalid move")
    return x, y


def format_move(move: Tuple[int, int], grid_size: int) -> str:
    x, y = move
    return f"{string.ascii_lowercase[grid_size - 1 - y]}{x + 1}"


class GTPAgent(Agent[HexState, HexAction]):
    def __init__(self, grid_size: int) -> None:
        self.process = pexpect.spawn(
            "/mnt/d/OneDrive - NTNU/NTNU/IT3105/Deep MCTS/venv/bin/python",
            [
                "/mnt/d/OneDrive - NTNU/NTNU/IT3105/Deep MCTS/deep_mcts/hex/gtp_interface.py"
            ],
            env={"PYTHONPATH": "/mnt/d/OneDrive - NTNU/NTNU/IT3105/Deep MCTS/"},
            encoding="utf-8",
        )
        self.process.sendline(f"boardsize {grid_size}")
        self.process.expect("= \r\n\r\n")
        self.game_manager = HexManager(grid_size)
        self.state = self.game_manager.initial_game_state()
        self.grid_size = grid_size

    def play(self, state: HexState) -> HexAction:
        if self.state != state:
            action = next(
                action
                for action, child_state in self.game_manager.generate_child_states(
                    self.state
                ).items()
                if child_state == state
            )
            move = format_move(action.coordinate, self.grid_size)
            self.process.sendline(f"play {format_player(self.state.player)} {move}")
            self.process.expect("= \r\n\r\n")
            self.state = state
        self.process.sendline(f"genmove {format_player(self.state.player)}")
        self.process.expect("= ([a-z]\\d{1,2})\r\n\r\n")
        move = self.process.match.group(1)
        x, y = parse_move(move, self.grid_size)
        action = HexAction((x, y))
        self.state = self.game_manager.generate_child_state(self.state, action)
        return action

    def reset(self) -> None:
        self.process.sendline("clear_board")
        self.process.expect("= \r\n\r\n")
        self.state = self.game_manager.initial_game_state()


if __name__ == "__main__":
    gtp_interface = GTPInterface()
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
