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


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class GameManager(ABC, Generic[S, A]):
    @abstractmethod
    def initial_game_state(self) -> S:
        ...

    @abstractmethod
    def generate_child_states(self, parent: S) -> Dict[A, S]:
        ...

    @abstractmethod
    def generate_child_state(self, parent: S, action: A) -> S:
        ...

    @abstractmethod
    def legal_actions(self, state: S) -> List[A]:
        ...

    @abstractmethod
    def is_final_state(self, state: S) -> bool:
        ...

    @abstractmethod
    def evaluate_final_state(self, state: S) -> float:
        ...
