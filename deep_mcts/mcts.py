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


class Node(Generic[_S]):
    __slots__ = ["state", "children", "E", "N", "P"]
    state: _S
    children: Dict[Action, "Node[_S]"]
    E: float
    N: int
    P: float

    def __init__(self, state: _S, P: float = -1.0) -> None:
        self.state = state
        self.children = {}
        self.E = 0
        self.N = 0
        self.P = P

    def u(self, parent: "Node[_S]") -> float:
        assert self.P != -1.0
        c = 1
        return c * self.P * sqrt(parent.N) / (1 + self.N)

    @property
    def Q(self) -> float:
        # Count unvisited nodes as lost
        if self.N == 0:
            assert self.E == 0
            return self.state.player.loss().value  # type: ignore[no-any-return]
        return self.E / self.N


class MCTS(Generic[_S]):
    game_manager: GameManager[_S]
    root: Node[_S]
    num_simulations: int
    rollout_policy: Optional[RolloutPolicy[_S]]
    state_evaluator: Optional[StateEvaluator[_S]]
    sample_move_cutoff: int
    dirichlet_alpha: float
    dirichlet_factor: float

    def __init__(
        self,
        game_manager: GameManager[_S],
        num_simulations: int,
        rollout_policy: Optional[RolloutPolicy[_S]],
        state_evaluator: Optional[StateEvaluator[_S]],
        sample_move_cutoff: int = 0,
        dirichlet_alpha: float = 0.0,
        dirichlet_factor: float = 0.25,
    ) -> None:
        self.game_manager = game_manager
        self.num_simulations = num_simulations
        self.rollout_policy = rollout_policy
        self.state_evaluator = state_evaluator
        initial_state = self.game_manager.initial_game_state()
        self.root = Node(initial_state)
        self.sample_move_cutoff = sample_move_cutoff
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_factor = dirichlet_factor

    def tree_search(self) -> List[Node[_S]]:
        path = [self.root]
        node = self.root
        while node.children:
            if node.state.player == Player.max_player():
                node = max(node.children.values(), key=lambda n: n.Q + n.u(node))
            else:
                node = min(node.children.values(), key=lambda n: n.Q - n.u(node))
            path.append(node)
        return path

    def expand_node(self, node: Node[_S]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        child_states = self.game_manager.generate_child_states(node.state)
        probabilities: Sequence[float]
        if self.state_evaluator is None:
            value, probabilities = (
                0.5,
                [
                    1 / len(child_states) if action in child_states else 0.0
                    for action in range(self.game_manager.num_actions)
                ],
            )
        else:
            value, probabilities = self.state_evaluator(node.state)
            assert len(probabilities) == self.game_manager.num_actions
        node.children = {
            action: Node(child_state, probabilities[action])
            for action, child_state in child_states.items()
        }

        return value

    def rollout(self, node: Node[_S]) -> float:
        assert (node.E, node.N) == (0.0, 0)
        if self.rollout_policy is None:
            return 0.5
        state = node.state
        while not self.game_manager.is_final_state(state):
            action = self.rollout_policy(state)
            state = self.game_manager.generate_child_state(state, action)
        return self.game_manager.evaluate_final_state(  # type: ignore[no-any-return]
            state
        ).value

    def backpropagate(self, path: Iterable[Node[_S]], evaluation: float) -> None:
        for node in path:
            node.N += 1
            node.E += evaluation
            assert 0.0 <= node.Q <= 1.0

    def evaluate_leaf(self, leaf_node: Node[_S]) -> float:
        if self.game_manager.is_final_state(leaf_node.state):
            return self.game_manager.evaluate_final_state(  # type: ignore[no-any-return]
                leaf_node.state
            ).value
        value = self.expand_node(leaf_node)
        rollout_value = self.rollout(leaf_node)
        if self.state_evaluator is None:
            return rollout_value
        elif self.rollout_policy is None:
            return value
        else:
            return (rollout_value + value) / 2

    def self_play(self) -> Iterable[Tuple[_S, _S, Action, Sequence[float]]]:
        i = 0
        while not self.game_manager.is_final_state(self.root.state):
            action_probabilities = self.step()
            if i < self.sample_move_cutoff:
                action = np.random.choice(
                    len(action_probabilities), p=action_probabilities
                )
            else:
                action = np.argmax(action_probabilities)
            next_node = self.root.children[action]
            current_node = self.root
            self.root = next_node
            i += 1
            yield current_node.state, next_node.state, action, action_probabilities

    def add_dirichlet_noise(self, node: Node[_S]) -> None:
        if self.dirichlet_alpha == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.children))
        for i, child in enumerate(node.children.values()):
            child.P = (
                child.P * (1 - self.dirichlet_factor) + noise[i] * self.dirichlet_factor
            )

    def step(self) -> List[float]:
        if not self.root.children:
            self.expand_node(self.root)
            self.root.N += 1
        self.add_dirichlet_noise(self.root)
        for _ in range(self.num_simulations):
            path = self.tree_search()
            leaf_node = path[-1]
            evaluation = self.evaluate_leaf(leaf_node)
            self.backpropagate(path, evaluation)
            if __debug__:
                if self.game_manager.is_final_state(leaf_node.state):
                    assert leaf_node.Q == evaluation
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
        self.root = Node(self.game_manager.initial_game_state())


class MCTSAgent(Agent[_S], ABC):
    mcts: MCTS[_S]
    epsilon: float

    def __init__(self, mcts: MCTS[_S], epsilon: float = 0.0) -> None:
        self.mcts = mcts
        self.epsilon = epsilon

    def play(self, state: _S) -> Action:
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
        action_probabilities = self.mcts.step()
        action: Action
        if self.epsilon > 0 and random.random() < self.epsilon:
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
        else:
            action = np.argmax(action_probabilities)
        self.mcts.root = self.mcts.root.children[action]
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
