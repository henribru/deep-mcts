from functools import lru_cache
from typing import List, Sequence

from deep_mcts.game import Player, CellState
from deep_mcts.hex.game import HexManager, HexState, Action
from deep_mcts.mcts import play_random_mcts


class HexWithSwapManager(HexManager):
    swap_move: Action

    def __init__(self, grid_size: int):
        super().__init__(grid_size, num_actions=grid_size ** 2 + 1)
        self.swap_move = grid_size ** 2

    @lru_cache(maxsize=2 ** 20)
    def generate_child_state(  # type: ignore[override]
        self, state: HexState, action: Action
    ) -> HexState:
        assert action in self.legal_actions(state)
        if action == self.swap_move:
            return HexState(
                state.player.opposite(),
                tuple(tuple(cell.opposite() for cell in row) for row in state.grid),
            )
        return super().generate_child_state(state, action)

    @lru_cache(maxsize=2 ** 20)
    def legal_actions(  # type: ignore[override]
        self, state: HexState
    ) -> List[Action]:
        actions = super().legal_actions(state)
        if (
            state.player == Player.SECOND
            and sum(sum(x != CellState.EMPTY for x in row) for row in state.grid) == 1
        ):
            actions.append(self.swap_move)
        return actions

    def probabilities_grid(self, action_probabilities: Sequence[float]) -> str:
        return f"{super().probabilities_grid(action_probabilities[:-1])}\nswap: {action_probabilities[-1]}"

    def action_str(self, action: Action) -> str:
        if action == self.swap_move:
            return "swap"
        return super().action_str(action)


if __name__ == "__main__":
    play_random_mcts(HexWithSwapManager(grid_size=5), num_simulations=1000)
