from abc import abstractmethod, ABC
from enum import IntEnum, Enum
from functools import lru_cache
from typing import TypeVar, Dict, Generic, List

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


# We can't make this (or any other state or action)
# frozen because frozen dataclasses with slots
# can't be pickled and transferred between processes.
# They should never be mutated though, so we enable
# hashing.
@dataclass(unsafe_hash=True)
class State:
    __slots__ = ["player"]
    player: Player


_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")


class GameManager(ABC, Generic[_S, _A]):
    @abstractmethod
    def initial_game_state(self) -> _S:
        ...

    @lru_cache(maxsize=2 ** 20)
    def generate_child_states(self, state: _S) -> Dict[_A, _S]:
        child_states = {
            action: self.generate_child_state(state, action)
            for action in self.legal_actions(state)
        }
        assert set(child_states.keys()) == set(self.legal_actions(state))
        return child_states

    @abstractmethod
    def generate_child_state(self, state: _S, action: _A) -> _S:
        ...

    @abstractmethod
    def legal_actions(self, state: _S) -> List[_A]:
        ...

    @abstractmethod
    def is_final_state(self, state: _S) -> bool:
        ...

    @abstractmethod
    def evaluate_final_state(self, state: _S) -> Outcome:
        ...
