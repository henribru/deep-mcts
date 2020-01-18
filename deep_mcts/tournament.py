import itertools
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Sequence, Generic, TYPE_CHECKING

from deep_mcts.game import State, Outcome, Action

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")
AgentComparison = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

if TYPE_CHECKING:
    from deep_mcts.mcts import GameManager


class Agent(ABC, Generic[_S]):
    @abstractmethod
    def play(self, state: _S) -> Action:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class RandomAgent(Agent[_S]):
    manager: "GameManager[_S]"

    def __init__(self, manager: "GameManager[_S]"):
        self.manager = manager

    def play(self, state: _S) -> Action:
        return random.choice(self.manager.legal_actions(state))

    def reset(self) -> None:
        pass


def tournament(
    agents: Sequence[Agent[_S]], num_games: int, game_manager: "GameManager[_S]"
) -> List[List[AgentComparison]]:
    results = [
        [((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)) for _ in range(len(agents))]
        for _ in range(len(agents))
    ]
    for (i, agent_1), (j, agent_2) in itertools.combinations_with_replacement(
        enumerate(agents), 2
    ):
        result = compare_agents((agent_1, agent_2), num_games, game_manager)
        results[i][j] = result
        results[j][i] = (result[2], result[1], result[0])
    return results


def compare_agents(
    players: Tuple[Agent[_S], Agent[_S]],
    num_games: int,
    game_manager: "GameManager[_S]",
) -> AgentComparison:
    first_player_wins = 0
    second_player_wins = 0
    first_player_draws = 0
    second_player_draws = 0
    first_player_losses = 0
    second_player_losses = 0
    for k in range(num_games):
        if k % 2 == 0:
            result = play(players, game_manager)
            if result == Outcome.FIRST_PLAYER_WIN:
                first_player_wins += 1
            elif result == Outcome.SECOND_PLAYER_WIN:
                first_player_losses += 1
            else:
                first_player_draws += 1
        else:
            result = play((players[1], players[0]), game_manager)
            if result == Outcome.SECOND_PLAYER_WIN:
                second_player_wins += 1
            elif result == Outcome.FIRST_PLAYER_WIN:
                second_player_losses += 1
            else:
                second_player_draws += 1
        # print(
        #     k + 1,
        #     (first_player_wins + second_player_wins) / ((k + 1)),
        #     2 * first_player_wins / (k + 1),
        #     2 * second_player_wins / (k + 1),
        # )
    num_games //= 2
    return (
        (first_player_wins / num_games, second_player_wins / num_games),
        (first_player_draws / num_games, second_player_draws / num_games),
        (first_player_losses / num_games, second_player_losses / num_games),
    )


def play(
    players: Tuple[Agent[_S], Agent[_S]], game_manager: "GameManager[_S]"
) -> Outcome:
    state = game_manager.initial_game_state()
    player = 0
    while not game_manager.is_final_state(state):
        action = players[player].play(state)
        state = game_manager.generate_child_state(state, action)
        player = (player + 1) % 2
    for player in players:
        player.reset()
    return game_manager.evaluate_final_state(state)
