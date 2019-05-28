import itertools
from typing import Tuple, List

from mcts import State, Action, StateManager, BehaviorPolicy


def topp(agents: List[BehaviorPolicy], num_games: int, state_manager: StateManager):
    results = [[0] * len(agents) for _ in range(len(agents))]
    for (i, agent_1), (j, agent_2) in itertools.combinations(enumerate(agents), 2):
        for k in range(num_games):
            if k % 2 == 0:
                players = (agent_1, agent_2)
                result = compare_agents(players, state_manager)
                results[i][j] += result
                results[j][i] -= result
            else:
                players = (agent_2, agent_1)
                result = compare_agents(players, state_manager)
                results[i][j] -= result
                results[j][i] += result
        results[i][j] = (results[i][j] + num_games) / (2 * num_games)
        results[j][i] = (results[j][i] + num_games) / (2 * num_games)
    return results


def compare_agents(
    players: Tuple[BehaviorPolicy, BehaviorPolicy], state_manager: StateManager
):
    state = state_manager.initial_game_state()
    player = 0
    while not state_manager.is_final_state(state):
        action = players[player](state)
        state = state_manager.generate_child_state(state, action)
        player = (player + 1) % 2
    return state_manager.evaluate_final_state(state)
