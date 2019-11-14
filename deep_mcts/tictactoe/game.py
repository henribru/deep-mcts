import random
from functools import lru_cache
from typing import Iterable, List, Tuple

from dataclasses import dataclass

from deep_mcts.game import CellState, GameManager, Player, State, Outcome
from deep_mcts.mcts import MCTS, play_random_mcts


@dataclass(unsafe_hash=True)
class TicTacToeState(State):
    __slots__ = ["grid"]
    grid: Tuple[Tuple[CellState, ...], ...]

    def __str__(self) -> str:
        cell_to_str = {
            CellState.EMPTY: "#",
            CellState.FIRST_PLAYER: "X",
            CellState.SECOND_PLAYER: "O",
        }
        return "\n".join("".join(cell_to_str[c] for c in row) for row in self.grid)


@dataclass(unsafe_hash=True)
class TicTacToeAction:
    __slots__ = ["coordinate"]
    coordinate: Tuple[int, int]


class TicTacToeManager(GameManager[TicTacToeState, TicTacToeAction]):
    def initial_game_state(self) -> TicTacToeState:
        return TicTacToeState(
            Player.FIRST,
            tuple(tuple(CellState.EMPTY for _ in range(3)) for _ in range(3)),
        )

    @lru_cache(maxsize=2 ** 20)
    def generate_child_state(  # type: ignore[override]
        self, state: TicTacToeState, action: TicTacToeAction
    ) -> TicTacToeState:
        assert action in self.legal_actions(state)
        x, y = action.coordinate
        return TicTacToeState(
            state.player.opposite(),
            tuple(
                tuple(
                    CellState(state.player) if (j, i) == (x, y) else cell
                    for j, cell in enumerate(row)
                )
                if i == y
                else row
                for i, row in enumerate(state.grid)
            ),
        )

    @lru_cache(maxsize=2 ** 20)
    def legal_actions(  # type: ignore[override]
        self, state: TicTacToeState
    ) -> List[TicTacToeAction]:
        return [
            TicTacToeAction((x, y))
            for y in range(3)
            for x in range(3)
            if state.grid[y][x] == CellState.EMPTY
        ]

    @lru_cache(maxsize=2 ** 20)
    def is_final_state(self, state: TicTacToeState) -> bool:  # type: ignore[override]
        return self.evaluate_final_state(state) != Outcome.DRAW or all(
            all(p != CellState.EMPTY for p in row) for row in state.grid
        )

    @lru_cache(maxsize=2 ** 20)
    def evaluate_final_state(  # type: ignore[override]
        self, state: TicTacToeState
    ) -> int:
        def rows_filled(player: CellState) -> bool:
            return any(all(p == player for p in state.grid[y]) for y in range(3))

        def columns_filled(player: CellState) -> bool:
            return any(
                all(state.grid[y][x] == player for y in range(3)) for x in range(3)
            )

        def diagonals_filled(player: CellState) -> bool:
            return all(state.grid[i][i] == player for i in range(3)) or all(
                state.grid[i][2 - i] == player for i in range(3)
            )

        for player, outcome in [
            (CellState.SECOND_PLAYER, Outcome.SECOND_PLAYER_WIN),
            (CellState.FIRST_PLAYER, Outcome.FIRST_PLAYER_WIN),
        ]:
            if (
                rows_filled(player)
                or columns_filled(player)
                or diagonals_filled(player)
            ):
                return outcome
        return Outcome.DRAW


if __name__ == "__main__":
    play_random_mcts(TicTacToeManager(), num_simulations=1000)
