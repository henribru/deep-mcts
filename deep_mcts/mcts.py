import random
from abc import abstractmethod, ABC
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
import numpy as np
import typing

from deep_mcts.game import GameManager, State, Action
from deep_mcts.tournament import Agent

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


class Node(Generic[_S, _A]):
    state: _S
    children: Dict[_A, "Node[_S, _A]"]
    E: float
    N: int
    P: float

    def __init__(self, state: _S) -> None:
        self.state = state
        self.children = {}
        self.E = 0
        self.N = 0
        self.P = 1

    def u(self, parent: "Node[_S, _A]") -> float:
        c = 1
        return c * self.P * sqrt(parent.N) / (1 + self.N)

    @property
    def Q(self) -> float:
        if self.N == 0:
            assert self.E == 0
            return 0
        return self.E / self.N


class MCTS(Generic[_S, _A]):
    game_manager: GameManager[_S, _A]
    root: Node[_S, _A]
    num_simulations: int
    rollout_policy: Optional[Callable[[_S], _A]]
    state_evaluator: Optional[Callable[[_S], Tuple[float, Dict[_A, float]]]]
    epsilon: float

    def __init__(
        self,
        game_manager: GameManager[_S, _A],
        num_simulations: int,
        rollout_policy: Optional[Callable[[_S], _A]],
        state_evaluator: Optional[Callable[[_S], Tuple[float, Dict[_A, float]]]],
        epsilon: float = 0,
    ) -> None:
        self.game_manager = game_manager
        self.num_simulations = num_simulations
        self.rollout_policy = rollout_policy
        self.state_evaluator = state_evaluator
        initial_state = self.game_manager.initial_game_state()
        self.root = Node(initial_state)
        self.epsilon = epsilon

    def tree_search(self) -> List[Node[_S, _A]]:
        path = [self.root]
        node = self.root
        while node.children:
            if node.state.player == 0:
                node = max(node.children.values(), key=lambda n: n.Q + n.u(node))
            else:
                node = min(node.children.values(), key=lambda n: n.Q - n.u(node))
            path.append(node)
        return path

    def expand_node(self, node: Node[_S, _A]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        child_states = self.game_manager.generate_child_states(node.state)
        node.children = {
            action: Node(child_state) for action, child_state in child_states.items()
        }
        if self.state_evaluator is None:
            value, probabilities = (
                0.0,
                {action: 1 / len(node.children) for action in node.children.keys()},
            )
        else:
            value, probabilities = self.state_evaluator(node.state)
        for action, node in node.children.items():
            node.P = probabilities[action]
        return value

    def rollout(self, node: Node[_S, _A]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        if self.rollout_policy is None:
            return 0
        state = node.state
        while not self.game_manager.is_final_state(state):
            action = self.rollout_policy(state)
            state = self.game_manager.generate_child_state(state, action)
        return self.game_manager.evaluate_final_state(state)

    def backpropagate(self, path: Iterable[Node[_S, _A]], evaluation: float) -> None:
        for node in path:
            node.N += 1
            node.E += evaluation
            assert -1.0 <= node.Q <= 1.0

    def evaluate_leaf(self, leaf_node: Node[_S, _A]) -> float:
        if self.game_manager.is_final_state(leaf_node.state):
            return self.game_manager.evaluate_final_state(leaf_node.state)
        else:
            return self.expand_node(leaf_node) + self.rollout(leaf_node)

    def self_play(self) -> Iterable[Tuple[_S, _S, _A, Dict[_A, float]]]:
        while not self.game_manager.is_final_state(self.root.state):
            action_probabilities = self.step()
            if self.epsilon > 0 and random.random() < self.epsilon:
                action = list(action_probabilities.keys())[
                    typing.cast(
                        int,
                        np.random.choice(
                            len(action_probabilities),
                            p=list(action_probabilities.values()),
                        ),
                    )
                ]
            else:
                action = max(
                    action_probabilities.keys(), key=lambda a: action_probabilities[a]
                )
            next_node = self.root.children[action]
            current_node = self.root
            self.root = next_node
            yield current_node.state, next_node.state, action, action_probabilities

    def step(self) -> Dict[_A, float]:
        for _ in range(self.num_simulations):
            path = self.tree_search()
            leaf_node = path[-1]
            evaluation = self.evaluate_leaf(leaf_node)
            self.backpropagate(path, evaluation)
            if self.game_manager.is_final_state(leaf_node.state):
                assert leaf_node.Q == evaluation
            else:
                assert (leaf_node.N, leaf_node.E) == (1, evaluation)
        assert self.root.N == sum(node.N for node in self.root.children.values()) + 1
        return {
            action: node.N / (self.root.N - 1)
            for action, node in self.root.children.items()
        }

    def reset(self) -> None:
        self.root = Node(self.game_manager.initial_game_state())


class MCTSAgent(Agent[_S, _A], ABC):
    mcts: MCTS[_S, _A]

    def __init__(self, mcts: MCTS[_S, _A]) -> None:
        self.mcts = mcts

    def play(self, state: _S) -> _A:
        if not any(child.state == state for child in self.mcts.root.children.values()):
            assert len(self.mcts.root.children) == 0
        self.mcts.root = next(
            (
                child
                for child in self.mcts.root.children.values()
                if child.state == state
            ),
            Node(state),
        )
        # self.mcts.root = next((child for child in self.mcts.root.children.values() if child.state == state), None)
        action_probabilities = self.mcts.step()
        if self.mcts.epsilon > 0 and random.random() < self.mcts.epsilon:
            action = list(action_probabilities.keys())[
                typing.cast(
                    int,
                    np.random.choice(
                        len(action_probabilities), p=list(action_probabilities.values())
                    ),
                )
            ]
        else:
            action = max(
                action_probabilities.keys(), key=lambda a: action_probabilities[a]
            )
        self.mcts.root = self.mcts.root.children[action]
        return action

    def reset(self) -> None:
        self.mcts.reset()
