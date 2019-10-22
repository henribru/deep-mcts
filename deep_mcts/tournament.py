import itertools
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Sequence, Generic

from deep_mcts.game import State
from deep_mcts.mcts import GameManager

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")


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
    results = [
        [(0.0, 0.0, 0.0) for _ in range(len(agents))] for _ in range(len(agents))
    ]
    for (i, agent_1), (j, agent_2) in itertools.combinations(enumerate(agents), 2):
        result = compare_agents((agent_1, agent_2), num_games, game_manager)
        results[i][j] = result
        results[j][i] = (result[2], result[1], result[0])
    return results


def compare_agents(
    players: Tuple[Agent[_S, _A], Agent[_S, _A]],
    num_games: int,
    game_manager: GameManager[_S, _A],
) -> Tuple[float, float, float]:
    wins, draws, losses = 0, 0, 0
    for k in range(num_games):
        if k % 2 == 0:
            result = play(players, game_manager)
        else:
            result = -play((players[1], players[0]), game_manager)
        if result == -1:
            wins += 1
        elif result == 1:
            losses += 1
        else:
            draws += 1
    return wins / num_games, draws / num_games, losses / num_games


def play(
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
