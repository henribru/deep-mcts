from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import TypeVar, Dict, Generic, List


@dataclass(frozen=True)
class State(ABC):
    player: int


@dataclass(frozen=True)
class Action(ABC):
    pass


_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


class GameManager(ABC, Generic[_S, _A]):
    @abstractmethod
    def initial_game_state(self) -> _S:
        ...

    @abstractmethod
    def generate_child_states(self, state: _S) -> Dict[_A, _S]:
        ...

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
