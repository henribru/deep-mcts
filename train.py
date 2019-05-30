from __future__ import annotations

import functools
from collections import deque
from typing import Iterable, Tuple, Dict

from anet import ANET
from game import State, Action
from hex.convolutional import ConvolutionalHexANET
from mcts import MCTS, GameManager
from topp import topp


def train(
    anet: ANET,
    state_manager: GameManager,
    num_actual_games: int,
    num_search_games: int,
    save_interval: int,
) -> Iterable[Tuple[State, State, Action, Dict[Action, float]]]:
    replay_buffer = deque([], 100000)
    random_anet = ConvolutionalHexANET(grid_size=4)
    anet.save(f"anet-0.pth")
    for i in range(num_actual_games):
        print(i + 1)
        mcts = MCTS(state_manager, num_search_games, functools.partial(anet.greedy_policy, epsilon=0.10), anet.evaluate_state)
        examples = []
        for state, next_state, action, visit_distribution in mcts.run():
            examples.append((state, visit_distribution))
            yield state, next_state, action, visit_distribution
        winner = state.player
        for state, visit_distribution in examples:
            # We want the target value to be from the perspective of the current player in that state
            replay_buffer.append((state, visit_distribution, 1 if state.player == winner else -1))  # TODO?
        anet.train(replay_buffer)
        results = topp([functools.partial(random_anet.greedy_policy, epsilon=0.10), functools.partial(anet.greedy_policy, epsilon=0.10)], 25, state_manager)
        print(results)
        if (i + 1) % save_interval == 0:
            anet.save(f"anet-{i + 1}.pth")
