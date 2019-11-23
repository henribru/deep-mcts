from abc import abstractmethod, ABC
from enum import IntEnum, Enum
from functools import lru_cache
from typing import TypeVar, Dict, Generic, List, Sequence

from dataclasses import dataclass


class Player(IntEnum):
    FIRST = 1
    SECOND = 0

    def opposite(self) -> "Player":
        if self == Player.FIRST:
            return Player.SECOND
        else:
            return Player.FIRST

    def win(self) -> "Outcome":
        if self == Player.FIRST:
            return Outcome.FIRST_PLAYER_WIN
        else:
            return Outcome.SECOND_PLAYER_WIN

    def loss(self) -> "Outcome":
        if self == Player.SECOND:
            return Outcome.FIRST_PLAYER_WIN
        else:
            return Outcome.SECOND_PLAYER_WIN

    @staticmethod
    def max_player() -> "Player":
        return Player.SECOND

    @staticmethod
    def min_player() -> "Player":
        return Player.FIRST


class CellState(IntEnum):
    FIRST_PLAYER = Player.FIRST
    SECOND_PLAYER = Player.SECOND
    EMPTY = -1

    def opposite(self) -> "CellState":
        if self == CellState.EMPTY:
            return CellState.EMPTY
        elif self == CellState.FIRST_PLAYER:
            return CellState.SECOND_PLAYER
        else:
            return CellState.FIRST_PLAYER


class Outcome(Enum):
    FIRST_PLAYER_WIN = 1.0 if Player.max_player() == Player.FIRST else 0.0
    SECOND_PLAYER_WIN = 1.0 if Player.max_player() == Player.SECOND else 0.0
    DRAW = 0.5


# We can't make this (or any other state)
# frozen because frozen dataclasses with slots
# can't be pickled and transferred between processes.
# They should never be mutated though, so we can enable
# hashing.
@dataclass(unsafe_hash=True)
class State:
    __slots__ = ["player"]
    player: Player


_S = TypeVar("_S", bound=State)
Action = int


class GameManager(ABC, Generic[_S]):
    num_actions: int
    grid_size: int

    @abstractmethod
    def initial_game_state(self) -> _S:
        ...

    @abstractmethod
    def generate_child_state(self, state: _S, action: Action) -> _S:
        ...

    @abstractmethod
    def legal_actions(self, state: _S) -> List[Action]:
        ...

    @abstractmethod
    def is_final_state(self, state: _S) -> bool:
        ...

    @abstractmethod
    def evaluate_final_state(self, state: _S) -> Outcome:
        ...

    @abstractmethod
    def probabilities_grid(self, action_probabilities: Sequence[float]) -> str:
        ...

    def action_str(self, action: Action) -> str:
        x, y = action % self.grid_size, action // self.grid_size
        return str((x, y))
