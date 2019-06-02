from __future__ import annotations

import functools
import random
from collections import deque
from typing import (
    Iterable,
    Tuple,
    Dict,
    Optional,
    TypeVar,
    Callable,
    Mapping,
    MutableSequence,
)

from deep_mcts.game import State, Action, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS
from deep_mcts.topp import topp

S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


def train(
    game_net: GameNet[S, A],
    game_manager: GameManager[S, A],
    num_actual_games: int,
    num_search_games: int,
    save_interval: int,
    rollout_policy: Optional[Callable[[S], A]] = None,
    opponent: Optional[Callable[[S], A]] = None,
) -> Iterable[Tuple[S, S, A, Dict[A, float]]]:
    replay_buffer: MutableSequence[Tuple[S, Mapping[A, float], float]] = deque(
        [], 100_000
    )
    game_net.save(f"anet-0.pth")
    for i in range(num_actual_games):
        print(i + 1)
        mcts = MCTS(
            game_manager,
            num_search_games,
            rollout_policy,
            state_evaluator=game_net.evaluate_state,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.run():
            examples.append((state, visit_distribution))
            yield state, next_state, action, visit_distribution
        outcome = game_manager.evaluate_final_state(next_state)
        for state, visit_distribution in examples:
            # We want the target value to be from the perspective of the current player in that state
            replay_buffer.append(
                (state, visit_distribution, outcome if state.player == 0 else -outcome)
            )  # TODO?
        examples = random.sample(replay_buffer, min(512, len(replay_buffer)))
        game_net.train(examples)
        if opponent is not None:
            results = topp(
                [opponent, functools.partial(game_net.greedy_policy, epsilon=0.05)],
                20,
                game_manager,
            )
            print(results)
        if (i + 1) % save_interval == 0:
            game_net.save(f"anet-{i + 1}.pth")
