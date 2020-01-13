import random
from abc import ABC
from math import sqrt
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Iterable,
    TypeVar,
    Generic,
    Optional,
    Sequence,
)

import numpy as np

from deep_mcts.game import GameManager, State, Player, Action
from deep_mcts.tournament import Agent

_S = TypeVar("_S", bound=State)
StateEvaluator = Callable[[_S], Tuple[float, Sequence[float]]]
RolloutPolicy = Callable[[_S], Action]


class Node:
    __slots__ = ["children", "E", "N", "P"]
    children: Dict[Action, "Node"]
    E: float
    N: int
    P: float

    def __init__(self, P: float = -1.0) -> None:
        self.E = 0
        self.N = 0
        self.P = P
        self.children = {}

    def u(self, parent: "Node") -> float:
        assert self.P != -1.0
        c = 1.25
        return c * self.P * sqrt(parent.N) / (1 + self.N)

    def Q(self, state: _S) -> float:
        # Count unvisited nodes as lost
        Q: float
        if self.N == 0:
            assert self.E == 0
            Q = state.player.loss().value
        else:
            Q = self.E / self.N
        assert 0.0 <= Q <= 1.0
        return Q


class MCTS(Generic[_S]):
    game_manager: GameManager[_S]
    root: Node
    state: _S
    num_simulations: int
    rollout_policy: Optional[RolloutPolicy[_S]]
    state_evaluator: Optional[StateEvaluator[_S]]
    sample_move_cutoff: int
    dirichlet_alpha: float
    dirichlet_factor: float
    rollout_share: float

    def __init__(
        self,
        game_manager: GameManager[_S],
        num_simulations: int,
        rollout_policy: Optional[RolloutPolicy[_S]],
        state_evaluator: Optional[StateEvaluator[_S]],
        sample_move_cutoff: int = 0,
        dirichlet_alpha: float = 0.0,
        dirichlet_factor: float = 0.25,
        rollout_share: float = 1.0,
    ) -> None:
        if rollout_policy is None and state_evaluator is None:
            raise ValueError("Both rollout_policy and state_evaluator cannot be None")
        self.game_manager = game_manager
        self.num_simulations = num_simulations
        self.rollout_policy = rollout_policy
        self.state_evaluator = state_evaluator
        self.state = self.game_manager.initial_game_state()
        self.root = Node()
        self.sample_move_cutoff = sample_move_cutoff
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_factor = dirichlet_factor
        self.rollout_share = rollout_share

    def tree_search(self) -> Tuple[List[Node], _S]:
        path = [self.root]
        node = self.root
        state = self.state
        while node.children:
            if state.player == Player.max_player():
                action, node = max(
                    node.children.items(),
                    key=lambda a_n: a_n[1].Q(state) + a_n[1].u(node),
                )
            else:
                action, node = min(
                    node.children.items(),
                    key=lambda a_n: a_n[1].Q(state) - a_n[1].u(node),
                )
            path.append(node)
            state = self.game_manager.generate_child_state(state, action)
        return path, state

    def expand_node(self, node: Node, state: _S) -> float:
        assert (node.E, node.N) == (0.0, 0)
        legal_actions = set(self.game_manager.legal_actions(state))
        probabilities: Sequence[float]
        if self.state_evaluator is None:
            value, probabilities = (
                0.5,
                [
                    1 / len(legal_actions) if action in legal_actions else 0.0
                    for action in range(self.game_manager.num_actions)
                ],
            )
        else:
            value, probabilities = self.state_evaluator(state)
            assert len(probabilities) == self.game_manager.num_actions
        node.children = {
            action: Node(probabilities[action]) for action in legal_actions
        }

        return value

    def rollout(self, node: Node, state: _S) -> float:
        assert (node.E, node.N) == (0.0, 0)
        while not self.game_manager.is_final_state(state):
            action = self.rollout_policy(state)  # type: ignore[misc]
            state = self.game_manager.generate_child_state(state, action)
        return self.game_manager.evaluate_final_state(state).value  # type: ignore[no-any-return]

    def backpropagate(self, path: Iterable[Node], evaluation: float) -> None:
        for node in path:
            node.N += 1
            node.E += evaluation

    def evaluate_leaf(self, leaf_node: Node, state: _S) -> float:
        if self.game_manager.is_final_state(state):
            return self.game_manager.evaluate_final_state(state).value  # type: ignore[no-any-return]
        value = self.expand_node(leaf_node, state)
        if self.state_evaluator is None:
            rollout_value = self.rollout(leaf_node, state)
            return rollout_value
        elif self.rollout_policy is None or (
            self.rollout_share < 1.0 and random.random() >= self.rollout_share
        ):
            return value
        else:
            rollout_value = self.rollout(leaf_node, state)
            return (rollout_value + value) / 2

    def self_play(self) -> Iterable[Tuple[_S, _S, Action, Sequence[float]]]:
        i = 0
        while not self.game_manager.is_final_state(self.state):
            action_probabilities = self.step()
            if i < self.sample_move_cutoff:
                action = np.random.choice(
                    len(action_probabilities), p=action_probabilities
                )
            else:
                action = np.argmax(action_probabilities)
            next_node = self.root.children[action]
            self.root = next_node
            old_state = self.state
            self.state = self.game_manager.generate_child_state(self.state, action)
            i += 1
            yield old_state, self.state, action, action_probabilities

    def add_dirichlet_noise(self, node: Node) -> None:
        if self.dirichlet_alpha == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        for i, child in enumerate(node.children.values()):
            child.P = (
                child.P * (1 - self.dirichlet_factor) + noise[i] * self.dirichlet_factor
            )

    def step(self) -> List[float]:
        if not self.root.children:
            self.expand_node(self.root, self.state)
            self.root.N += 1
        self.add_dirichlet_noise(self.root)
        for _ in range(self.num_simulations):
            path, state = self.tree_search()
            leaf_node = path[-1]
            evaluation = self.evaluate_leaf(leaf_node, state)
            self.backpropagate(path, evaluation)
            if __debug__:
                if self.game_manager.is_final_state(state):
                    assert leaf_node.Q(state) == evaluation
                else:
                    assert (leaf_node.N, leaf_node.E) == (1, evaluation)
        visit_sum = sum(node.N for node in self.root.children.values())
        return [
            self.root.children[action].N / visit_sum
            if action in self.root.children
            else 0.0
            for action in range(self.game_manager.num_actions)
        ]

    def reset(self) -> None:
        self.root = Node()
        self.state = self.game_manager.initial_game_state()


class MCTSAgent(Agent[_S], ABC):
    mcts: MCTS[_S]
    epsilon: float

    def __init__(self, mcts: MCTS[_S], epsilon: float = 0.0) -> None:
        self.mcts = mcts
        self.epsilon = epsilon

    def play(self, state: _S) -> Action:
        if __debug__ and not any(
            self.mcts.game_manager.generate_child_state(self.mcts.state, action)
            == state
            for action, node in self.mcts.root.children.items()
        ):
            assert len(self.mcts.root.children) == 0
        self.mcts.root = next(
            (
                node
                for action, node in self.mcts.root.children.items()
                if self.mcts.game_manager.generate_child_state(self.mcts.state, action)
                == state
            ),
            Node(),
        )
        self.mcts.state = state
        action_probabilities = self.mcts.step()
        action: Action
        if self.epsilon > 0 and random.random() < self.epsilon:
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
        else:
            action = np.argmax(action_probabilities)
        self.mcts.root = self.mcts.root.children[action]
        self.mcts.state = self.mcts.game_manager.generate_child_state(state, action)
        return action

    def reset(self) -> None:
        self.mcts.reset()


def play_random_mcts(manager: GameManager[_S], num_simulations: int) -> None:
    mcts = MCTS(
        manager,
        num_simulations,
        lambda state: random.choice(manager.legal_actions(state)),
        None,
    )
    for state, next_state, action, _ in mcts.self_play():
        print(state)
        print()
        print(manager.action_str(action))
        print()
    print(next_state)
    print(state.player)
