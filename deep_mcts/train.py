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
    Deque,
    List,
    Dict,
)

from deep_mcts.game import State, Action, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS
from deep_mcts.topp import topp
from hex.convolutionalnet import ConvolutionalHexNet

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


def train(
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    save_interval: int,
    evaluation_interval: int,
    rollout_policy: Optional[Callable[[_S], _A]] = None,
) -> Iterable[Tuple[int, float, Optional[float]]]:
    replay_buffer = Deque[Tuple[_S, Dict[_A, float], float]]([], 100_000)
    game_net.save(f"anet-0.pth")
    random_opponent = lambda s: random.choice(game_manager.legal_actions(s))
    # random_opponent = functools.partial(
    #     ConvolutionalHexNet(game_net.grid_size).greedy_policy, epsilon=0.05
    # )
    original_opponent = game_net.copy()
    previous_opponent: Optional[GameNet[_S, _A]] = None
    for i in range(num_games):
        mcts = MCTS(
            game_manager,
            num_simulations,
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
            original_evaluation = (
                topp(
                    [
                        original_opponent.greedy_policy,
                        game_net.greedy_policy,
                    ],
                    2,
                    game_manager,
                )[1][0]
            )
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
            print(i + 1, random_evaluation, previous_evaluation, original_evaluation)
            yield i + 1, random_evaluation, previous_evaluation
            previous_opponent = game_net.copy()
        if (i + 1) % save_interval == 0:
            filename = f"anet-{i + 1}.pth"
            game_net.save(filename)
            print(f"Saved {filename}")
