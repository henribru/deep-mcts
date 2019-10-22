from abc import abstractmethod, ABC
from enum import IntEnum

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


class Outcome(IntEnum):
    FIRST_PLAYER_WIN = -1
    SECOND_PLAYER_WIN = 1
    DRAW = 0


@dataclass(frozen=True)
class State:
    player: Player


_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")


class GameManager(ABC, Generic[_S, _A]):
    @abstractmethod
    def initial_game_state(self) -> _S:
        ...

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
    def evaluate_final_state(self, state: _S) -> int:
        ...
