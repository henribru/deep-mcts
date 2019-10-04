from __future__ import annotations

import itertools
from typing import Tuple, List, Callable, TypeVar, Sequence

from deep_mcts.game import State, Action
from deep_mcts.mcts import GameManager

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


def topp(
    agents: Sequence[Callable[[_S], _A]], num_games: int, game_manager: GameManager[_S, _A]
) -> List[List[float]]:
    results = [[0.0] * len(agents) for _ in range(len(agents))]
    for (i, agent_1), (j, agent_2) in itertools.combinations(enumerate(agents), 2):
        for k in range(num_games):
            if k % 2 == 0:
                players = (agent_1, agent_2)
                result = compare_agents(players, game_manager)
                results[i][j] += result
                results[j][i] -= result
            else:
                players = (agent_2, agent_1)
                result = compare_agents(players, game_manager)
                results[i][j] -= result
                results[j][i] += result
        results[i][j] = (results[i][j] + num_games) / (2 * num_games)
        results[j][i] = (results[j][i] + num_games) / (2 * num_games)
    return results


def compare_agents(
    players: Tuple[Callable[[_S], _A], Callable[[_S], _A]], game_manager: GameManager[_S, _A]
) -> int:
    state = game_manager.initial_game_state()
    player = 0
    while not game_manager.is_final_state(state):
        action = players[player](state)
        state = game_manager.generate_child_state(state, action)
        player = (player + 1) % 2
    return game_manager.evaluate_final_state(state)
