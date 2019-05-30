from __future__ import annotations

from math import sqrt
from typing import Callable, Dict, List, Tuple, Iterable, TypeVar, Generic

from game import GameManager, State, Action

S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class Node(Generic[S, A]):
    state: S
    children: Dict[A, Node[S, A]]
    E: int
    N: int
    P: float

    def __init__(self, state: S):
        self.state = state
        self.children = {}
        self.E = 0
        self.N = 0
        self.P = 0

    def u(self, parent: Node[S, A]) -> float:
        c = 1
        return c * self.P * sqrt(parent.N) / (1 + self.N)

    @property
    def Q(self) -> float:
        # TODO?
        if self.N == 0:
            return 0
        return self.E / self.N


class MCTS(Generic[S, A]):
    state_manager: GameManager[S, A]
    root: Node[S, A]
    M: int
    behavior_policy: Callable[[S], A]
    state_evaluator: Callable[[S], Tuple[float, Dict[A, float]]]

    def __init__(
            self, state_manager: GameManager[S, A], M: int, behavior_policy: Callable[[S], A],
            state_evaluator: Callable[[S], Tuple[float, Dict[A, float]]]):
        self.state_manager = state_manager
        self.M = M
        self.behavior_policy = behavior_policy
        self.state_evaluator = state_evaluator
        initial_state = self.state_manager.initial_game_state()
        self.root = Node(initial_state)

    def tree_search(self) -> List[Node[S, A]]:
        path = [self.root]
        node = self.root
        while node.children:
            if node.state.player == 0:
                node = max(node.children.values(), key=lambda n: n.Q + n.u(node))
            else:
                node = min(node.children.values(), key=lambda n: n.Q - n.u(node))
            path.append(node)
        return path

    def expand_node(self, node: Node[S, A]) -> None:
        child_states = self.state_manager.generate_child_states(node.state)
        node.children = {action: Node(child_state) for action, child_state in child_states.items()}
        value, probabilities = self.state_evaluator(node.state)
        node.E = value
        for action, node in node.children.items():
            node.P = probabilities[action]

    def evaluate_leaf(self, leaf_node: Node[S, A], rollout=False) -> float:
        if not rollout:
            return leaf_node.E
        state = leaf_node.state
        while not self.state_manager.is_final_state(state):
            action = self.behavior_policy(state)
            state = self.state_manager.generate_child_state(state, action)
        return self.state_manager.evaluate_final_state(state)

    def backpropagate(self, path: List[Node[S, A]], evaluation: float) -> None:
        for node in path:
            node.N += 1
            node.E += evaluation

    def run(self) -> Iterable[Tuple[S, S, A, Dict[A, float]]]:
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
