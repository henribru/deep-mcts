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
    Iterator,
)

from deep_mcts.game import State, Action, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS
from deep_mcts.topp import topp
from hex.convolutionalnet import ConvolutionalHexNet

S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


def train(
    game_net: GameNet[S, A],
    game_manager: GameManager[S, A],
    num_actual_games: int,
    num_search_games: int,
    save_interval: int,
    evaluation_interval: int,
    rollout_policy: Optional[Callable[[S], A]] = None,
) -> Iterable[Tuple[int, float, float]]:
    replay_buffer: MutableSequence[Tuple[S, Mapping[A, float], float]] = deque(
        [], 100_000
    )
    game_net.save(f"anet-0.pth")
    random_opponent = lambda s: random.choice(game_manager.legal_actions(s))
    # random_opponent = functools.partial(
    #     ConvolutionalHexNet(game_net.grid_size).greedy_policy, epsilon=0.05
    # )
    previous_opponent: Optional[GameNet[S, A]] = None
    for i in range(num_actual_games):
        mcts = MCTS(
            game_manager,
            num_search_games,
            rollout_policy,
            state_evaluator=game_net.evaluate_state,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.run():
            examples.append((state, visit_distribution))
        outcome = game_manager.evaluate_final_state(next_state)
        for state, visit_distribution in examples:
            # We want the target value to be from the perspective of the current player in that state
            replay_buffer.append(
                (state, visit_distribution, outcome if state.player == 0 else -outcome)
            )  # TODO?
        examples = random.sample(replay_buffer, min(512, len(replay_buffer)))
        game_net.train(examples)
        if (i + 1) % evaluation_interval == 0:
            random_evaluation = topp(
                [random_opponent, game_net.greedy_policy], 20, game_manager
            )[1][0]
            previous_evaluation = (
                topp(
                    [
                        functools.partial(
                            previous_opponent.greedy_policy, epsilon=0.05
                        ),
                        functools.partial(game_net.greedy_policy, epsilon=0.05),
                    ],
                    20,
                    game_manager,
                )[1][0]
                if previous_opponent is not None
                else None
            )
            print(i + 1, random_evaluation, previous_evaluation)
            yield i + 1, random_evaluation, previous_evaluation
            previous_opponent = game_net.copy()
        if (i + 1) % save_interval == 0:
            game_net.save(f"anet-{i + 1}.pth")
