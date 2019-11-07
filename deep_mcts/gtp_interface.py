import dataclasses
import os.path
import re
import string
import sys
from typing import (
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Generic,
    Mapping,
)
from abc import ABC, abstractmethod
import pexpect

from deep_mcts.game import Player, Outcome, State, GameManager
from deep_mcts.gamenet import GameNet, DEVICE
from deep_mcts.mcts import MCTS, Node
from deep_mcts.tournament import Agent


_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")


class GTPInterface(ABC, Generic[_S, _A]):
    commands: Dict[str, Callable[[List[str]], Optional[str]]]
    game_manager: GameManager[_S, _A]
    mcts: MCTS[_S, _A]
    board_size: int

    def __init__(self, board_size: int) -> None:
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
            "result": self.result,
        }
        self.board_size = board_size
        self.game_manager = self.get_game_manager(board_size)
        self.net = self.get_game_net(board_size).to(DEVICE)
        self.mcts = MCTS(
            self.game_manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=self.net.evaluate_state,
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
        self.game_manager = self.get_game_manager(self.board_size)
        self.net = self.get_game_net(self.board_size)
        self.mcts = MCTS(
            self.game_manager,
            num_simulations=100,
            rollout_policy=None,
            state_evaluator=self.net.evaluate_state,
        )

    def play(self, args: List[str]) -> None:
        if len(args) != 2:
            raise ValueError("play takes 2 arguments")
        player = self.parse_player(args[0])
        action = self.parse_move(args[1], self.board_size)
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
        player = self.parse_player(args[0])
        actual_player = self.mcts.root.state.player
        if actual_player != player:
            self.mcts.root = Node(
                dataclasses.replace(self.mcts.root.state, player=player)
            )
        action_probabilities = self.mcts.step()
        value, net_action_probabilities = self.net.evaluate_state(self.mcts.root.state)
        print(value, file=sys.stderr)
        print(
            self.probabilities_grid(net_action_probabilities, self.board_size),
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print(
            self.probabilities_grid(action_probabilities, self.board_size),
            file=sys.stderr,
        )
        action = max(action_probabilities.keys(), key=lambda a: action_probabilities[a])
        self.mcts.root = self.mcts.root.children[action]
        return self.format_move(action)

    def showboard(self, args: List[str]) -> str:
        return f"\n{self.mcts.root.state}"

    def result(self, args: List[str]) -> str:
        return str(self.game_manager.evaluate_final_state(self.mcts.root.state))

    @staticmethod
    def parse_player(player: str) -> int:
        player = player.lower()
        if player == "w":
            player = "white"
        elif player == "b":
            player = "black"
        if player not in ["white", "black"]:
            raise ValueError("invalid player")
        return Player.FIRST if player == "black" else Player.SECOND

    @staticmethod
    @abstractmethod
    def parse_move(move: str, board_size: int) -> _A:
        ...

    @staticmethod
    def format_player(player: int) -> str:
        return "black" if player == Player.FIRST else "white"

    @staticmethod
    @abstractmethod
    def format_move(move: _A) -> str:
        ...

    @staticmethod
    @abstractmethod
    def get_game_manager(board_size: int) -> GameManager[_S, _A]:
        ...

    @staticmethod
    @abstractmethod
    def get_game_net(board_size: int) -> GameNet[_S, _A]:
        ...

    @staticmethod
    @abstractmethod
    def probabilities_grid(
        action_probabilities: Mapping[_A, float], board_size: int
    ) -> str:
        ...


class GTPAgent(Agent[_S, _A]):
    def __init__(self, manager: GameManager[_S, _A], grid_size: int) -> None:
        self.process = pexpect.spawn(
            sys.executable,
            [os.path.dirname(__file__)],
            env={"PYTHONPATH": os.path.join(os.path.dirname(__file__), "..")},
            encoding="utf-8",
        )
        self.process.sendline(f"boardsize {grid_size}")
        self.process.expect("= \r\n\r\n")
        self.game_manager = manager
        self.state = self.game_manager.initial_game_state()
        self.grid_size = grid_size

    def play(self, state: _S) -> _A:
        if self.state != state:
            action = next(
                action
                for action, child_state in self.game_manager.generate_child_states(
                    self.state
                ).items()
                if child_state == state
            )
            move = GTPInterface.format_move(action)
            self.process.sendline(
                f"play {GTPInterface.format_player(self.state.player)} {move}"
            )
            self.process.expect("= \r\n\r\n")
            self.state = state
        self.process.sendline(
            f"genmove {GTPInterface.format_player(self.state.player)}"
        )
        self.process.expect("= (.+)\r\n\r\n")
        move = self.process.match.group(1)
        action = GTPInterface.parse_move(move, self.grid_size)
        self.state = self.game_manager.generate_child_state(self.state, action)
        return action

    def reset(self) -> None:
        self.process.sendline("clear_board")
        self.process.expect("= \r\n\r\n")
        self.state = self.game_manager.initial_game_state()
