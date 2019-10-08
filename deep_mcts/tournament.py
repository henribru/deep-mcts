import itertools
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Sequence, Generic

from deep_mcts.game import State, Action
from deep_mcts.mcts import GameManager

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


class Agent(ABC, Generic[_S, _A]):
    @abstractmethod
    def play(self, state: _S) -> _A:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class RandomAgent(Agent[_S, _A]):
    manager: GameManager[_S, _A]

    def __init__(self, manager: GameManager[_S, _A]):
        self.manager = manager

    def play(self, state: _S) -> _A:
        return random.choice(self.manager.legal_actions(state))

    def reset(self) -> None:
        pass


def tournament(
    agents: Sequence[Agent[_S, _A]], num_games: int, game_manager: GameManager[_S, _A]
) -> List[List[Tuple[float, float, float]]]:
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
    results = [[(w / num_games, d / num_games, l / num_games) for w, d, l in row] for row in results]
    return results


def compare_agents(
    players: Tuple[Agent[_S, _A], Agent[_S, _A]], game_manager: GameManager[_S, _A]
) -> int:
    state = game_manager.initial_game_state()
    player = 0
    while not game_manager.is_final_state(state):
        action = players[player].play(state)
        state = game_manager.generate_child_state(state, action)
        player = (player + 1) % 2
    for player in players:
        player.reset()
    return game_manager.evaluate_final_state(state)