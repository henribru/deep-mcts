from functools import lru_cache
from typing import List, Union, Mapping, cast

from dataclasses import dataclass

from deep_mcts.game import Player, CellState, GameManager
from deep_mcts.hex.game import HexManager, HexState, HexAction, hex_probabilities_grid
from deep_mcts.mcts import play_random_mcts


@dataclass(frozen=True)
class HexSwap:
    def __str__(self) -> str:
        return "swap"


HexWithSwapAction = Union[HexAction, HexSwap]


class HexWithSwapManager(GameManager[HexState, HexWithSwapAction]):
    manager: HexManager

    def __init__(self, grid_size: int):
        self.manager = HexManager(grid_size)

    def initial_game_state(self) -> HexState:
        return self.manager.initial_game_state()

    def evaluate_final_state(self, state: HexState) -> int:
        return self.manager.evaluate_final_state(state)

    def is_final_state(self, state: HexState) -> bool:
        return self.manager.is_final_state(state)

    @lru_cache(maxsize=2 ** 20)
    def generate_child_state(  # type: ignore[override]
        self, state: HexState, action: HexWithSwapAction
    ) -> HexState:
        assert action in self.legal_actions(state)
        if isinstance(action, HexSwap):
            return HexState(
                state.player.opposite(),
                tuple(
                    tuple(state.grid[j][i].opposite() for i in range(self.grid_size))
                    for j in range(self.grid_size)
                ),
            )
        return self.manager.generate_child_state(state, action)

    @lru_cache(maxsize=2 ** 20)
    def legal_actions(  # type: ignore[override]
        self, state: HexState
    ) -> List[HexWithSwapAction]:
        actions = cast(List[HexWithSwapAction], self.manager.legal_actions(state))
        if (
            state.player == Player.SECOND
            and sum(sum(x != CellState.EMPTY for x in row) for row in state.grid) == 1
        ):
            actions.append(HexSwap())
        return actions


def hex_with_swap_probabilities_grid(
    action_probabilities: Mapping[HexWithSwapAction, float], grid_size: int
) -> str:
    swap_probability = 0.0
    hex_action_probabilities = {}
    for action, probability in action_probabilities.items():
        if isinstance(action, HexSwap):
            swap_probability = probability
        else:
            hex_action_probabilities[action] = probability
    return f"{hex_probabilities_grid(hex_action_probabilities, grid_size)}\nswap: {swap_probability}"


if __name__ == "__main__":
    play_random_mcts(HexWithSwapManager(grid_size=5), num_simulations=1000)
