from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from math import log, sqrt
from typing import Callable, Dict, List, TypeVar, Tuple


@dataclass(frozen=True)
class State(ABC):
    player: int


@dataclass(frozen=True)
class Action(ABC):
    pass


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class StateManager(ABC):
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
    def is_final_state(self, state: S) -> bool:
        ...

    @abstractmethod
    def evaluate_final_state(self, state: S) -> float:
        ...


class Node:
    state: State
    children: Dict[Action, Node]
    E: int
    N: int
    P: float

    def __init__(self, state: State):
        self.state = state
        self.children = {}
        self.E = 0
        self.N = 0
        self.P = 0

    def u(self, parent: Node) -> float:
        c = 1
        return c * self.P * sqrt(parent.N) / (1 + self.N)

    @property
    def Q(self) -> float:
        # TODO?
        if self.N == 0:
            return 0
        return self.E / self.N


BehaviorPolicy = Callable[[State], Action]
StateEvaluator = Callable[[State], Tuple[float, Dict[Action, float]]]


class MCTS:
    state_manager: StateManager
    root: Node
    M: int
    behavior_policy: BehaviorPolicy

    def __init__(
            self, state_manager: StateManager, M: int, behavior_policy: BehaviorPolicy,
            state_evaluator: StateEvaluator):
        self.state_manager = state_manager
        self.M = M
        self.behavior_policy = behavior_policy
        self.state_evaluator = state_evaluator
        initial_state = self.state_manager.initial_game_state()
        self.root = Node(initial_state)

    def tree_search(self) -> List[Node]:
        path = [self.root]
        node = self.root
        while node.children:
            if node.state.player == 0:
                node = max(node.children.values(), key=lambda n: n.Q + n.u(node))
            else:
                node = min(node.children.values(), key=lambda n: n.Q - n.u(node))
            path.append(node)
        return path

    def expand_node(self, node: Node):
        child_states = self.state_manager.generate_child_states(node.state)
        node.children = {action: Node(child_state) for action, child_state in child_states.items()}
        value, probabilities = self.state_evaluator(node.state)
        node.E = value
        for action, node in node.children.items():
            node.P = probabilities[action]

    def evaluate_leaf(self, leaf_node: Node, rollout=False) -> float:
        if not rollout:
            return leaf_node.E
        state = leaf_node.state
        while not self.state_manager.is_final_state(state):
            action = self.behavior_policy(state)
            state = self.state_manager.generate_child_state(state, action)
        return self.state_manager.evaluate_final_state(state)

    def backpropagate(self, path: List[Node], evaluation: float):
        for node in path:
            node.N += 1
            node.E += evaluation

    def run(self):
        while not self.state_manager.is_final_state(self.root.state):
            for _ in range(self.M):
                path = self.tree_search()
                leaf_node = path[-1]
                if not self.state_manager.is_final_state(leaf_node.state):
                    self.expand_node(leaf_node)
                    path.append(next(iter(leaf_node.children.values())))
                evaluation = self.evaluate_leaf(path[-1])
                self.backpropagate(path, evaluation)
            action, next_node = max(self.root.children.items(), key=lambda c: c[1].N)
            yield self.root.state, next_node.state, action, {
                action: node.N / self.root.N
                for action, node in self.root.children.items()
            }
            self.root = next_node
