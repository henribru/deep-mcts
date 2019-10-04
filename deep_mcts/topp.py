import itertools
from typing import Tuple, List, Callable, TypeVar, Sequence

from deep_mcts.game import State, Action
from deep_mcts.mcts import GameManager

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


def topp(
    agents: Sequence[Callable[[_S], _A]], num_games: int, game_manager: GameManager[_S, _A]
) -> List[List[List[int]]]:
    results = [[[0, 0, 0] for _ in range(len(agents))] for _ in range(len(agents))]
    for (i, agent_1), (j, agent_2) in itertools.combinations(enumerate(agents), 2):
        for k in range(num_games):
            if k % 2 == 0:
                players = (agent_1, agent_2)
                result = compare_agents(players, game_manager)
                if result == 1:
                    results[i][j][0] += 1
                    results[j][i][2] += 1
                elif result == -1:
                    results[j][i][0] += 1
                    results[i][j][2] += 1
                else:
                    results[i][j][1] += 1
                    results[j][i][1] += 1
            else:
                players = (agent_2, agent_1)
                result = compare_agents(players, game_manager)
                if result == 1:
                    results[j][i][0] += 1
                    results[i][j][2] += 1
                elif result == -1:
                    results[i][j][0] += 1
                    results[j][i][2] += 1
                else:
                    results[i][j][1] += 1
                    results[j][i][1] += 1
        for k in range(3):
            results[i][j][k] /= num_games
            results[j][i][k] /= num_games
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
