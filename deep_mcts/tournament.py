import itertools
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Sequence, Generic, TYPE_CHECKING

from deep_mcts.game import State

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")

if TYPE_CHECKING:
    from deep_mcts.mcts import GameManager


class Agent(ABC, Generic[_S, _A]):
    @abstractmethod
    def play(self, state: _S) -> _A:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class RandomAgent(Agent[_S, _A]):
    manager: "GameManager[_S, _A]"

    def __init__(self, manager: "GameManager[_S, _A]"):
        self.manager = manager

    def play(self, state: _S) -> _A:
        return random.choice(self.manager.legal_actions(state))

    def reset(self) -> None:
        pass


def tournament(
    agents: Sequence[Agent[_S, _A]], num_games: int, game_manager: "GameManager[_S, _A]"
) -> List[List[Tuple[float, float, float]]]:
    results = [
        [(0.0, 0.0, 0.0) for _ in range(len(agents))] for _ in range(len(agents))
    ]
    for (i, agent_1), (j, agent_2) in itertools.combinations(enumerate(agents), 2):
        result = compare_agents((agent_1, agent_2), num_games, game_manager)
        results[i][j] = (sum(result[0]), sum(result[1]), sum(result[2]))
        results[j][i] = (sum(result[2]), sum(result[1]), sum(result[0]))
    return results


def compare_agents(
    players: Tuple[Agent[_S, _A], Agent[_S, _A]],
    num_games: int,
    game_manager: "GameManager[_S, _A]",
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    first_player_wins = 0
    second_player_wins = 0
    first_player_draws = 0
    second_player_draws = 0
    first_player_losses = 0
    second_player_losses = 0
    for k in range(num_games):
        if k % 2 == 0:
            result = play(players, game_manager)
            if result == -1:
                first_player_wins += 1
            elif result == 1:
                first_player_losses += 1
            else:
                first_player_draws += 1
        else:
            result = -play((players[1], players[0]), game_manager)
            if result == -1:
                second_player_wins += 1
            elif result == 1:
                second_player_losses += 1
            else:
                second_player_draws += 1
    num_games //= 2
    return (
        (first_player_wins / num_games, second_player_wins / num_games),
        (first_player_draws / num_games, second_player_draws / num_games),
        (first_player_losses / num_games, second_player_losses / num_games),
    )


def play(
    players: Tuple[Agent[_S, _A], Agent[_S, _A]], game_manager: "GameManager[_S, _A]"
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
