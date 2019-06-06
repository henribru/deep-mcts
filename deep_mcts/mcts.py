from __future__ import annotations

from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Iterable,
    TypeVar,
    Generic,
    Optional,
    Mapping,
)

from math import sqrt

from deep_mcts.game import GameManager, State, Action

S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class Node(Generic[S, A]):
    state: S
    children: Dict[A, Node[S, A]]
    E: float
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
    game_manager: GameManager[S, A]
    root: Node[S, A]
    M: int
    rollout_policy: Optional[Callable[[S], A]]
    state_evaluator: Callable[[S], Tuple[float, Mapping[A, float]]]

    def __init__(
        self,
        game_manager: GameManager[S, A],
        M: int,
        rollout_policy: Optional[Callable[[S], A]],
        state_evaluator: Callable[[S], Tuple[float, Mapping[A, float]]],
    ):
        self.game_manager = game_manager
        self.M = M
        self.rollout_policy = rollout_policy
        self.state_evaluator = state_evaluator
        initial_state = self.game_manager.initial_game_state()
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

    def expand_node(self, node: Node[S, A]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        child_states = self.game_manager.generate_child_states(node.state)
        node.children = {
            action: Node(child_state) for action, child_state in child_states.items()
        }
        value, probabilities = self.state_evaluator(node.state)
        for action, node in node.children.items():
            node.P = probabilities[action]
        return value

    def rollout(self, node: Node[S, A]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        if self.rollout_policy is None:
            return 0
        state = node.state
        while not self.game_manager.is_final_state(state):
            action = self.rollout_policy(state)
            state = self.game_manager.generate_child_state(state, action)
        return self.game_manager.evaluate_final_state(state)

    def backpropagate(self, path: Iterable[Node[S, A]], evaluation: float) -> None:
        for node in path:
            node.N += 1
            node.E += evaluation
            assert -1.0 <= node.Q <= 1.0

    def evaluate_leaf(self, leaf_node: Node[S, A]) -> float:
        if self.game_manager.is_final_state(leaf_node.state):
            return self.game_manager.evaluate_final_state(leaf_node.state)
        else:
            return self.expand_node(leaf_node) + self.rollout(leaf_node)

    def run(self) -> Iterable[Tuple[S, S, A, Dict[A, float]]]:
        while not self.game_manager.is_final_state(self.root.state):
            for _ in range(self.M):
                path = self.tree_search()
                leaf_node = path[-1]
                evaluation = self.evaluate_leaf(leaf_node)
                self.backpropagate(path, evaluation)
                if __debug__:
                    if self.game_manager.is_final_state(leaf_node.state):
                        assert leaf_node.Q == evaluation
                    else:
                        assert (leaf_node.N, leaf_node.E) == (1, evaluation)
            action, next_node = max(self.root.children.items(), key=lambda c: c[1].N)
            yield self.root.state, next_node.state, action, {
                action: node.N / self.root.N
                for action, node in self.root.children.items()
            }
            self.root = next_node
