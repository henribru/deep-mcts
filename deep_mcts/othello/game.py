import itertools
import string
from functools import lru_cache
from typing import Iterator, List, Tuple, Set, Sequence

import dataclasses
from dataclasses import dataclass

from deep_mcts.game import CellState, GameManager, Outcome, Action, Player, State
from deep_mcts.mcts import play_random_mcts


@dataclass(unsafe_hash=True)
class OthelloState(State):
    __slots__ = ["grid"]
    grid: Tuple[Tuple[CellState, ...], ...]

    def __str__(self) -> str:
        symbol = {-1: ".", 0: "0", 1: "1"}
        grid = []
        letters = string.ascii_uppercase[: len(self.grid)]
        width = 2 * len(self.grid) - 1 + 4
        grid.append(" ".join(letters).center(width))
        for i, row in enumerate(self.grid, 1):
            row_str = " ".join(symbol[cell] for cell in row)
            grid.append(f"{i} {row_str} {i}".center(width))
        grid.append(" ".join(letters).center(width))
        return "\n".join(grid)


class OthelloManager(GameManager[OthelloState]):
    pass_move: Action

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size ** 2 + 1
        self.pass_move = grid_size ** 2

    def initial_game_state(self) -> OthelloState:
        board = [
            [CellState.EMPTY for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]
        x, y = self.grid_size // 2, self.grid_size // 2
        board[y][x] = CellState.SECOND_PLAYER
        board[y][x - 1] = CellState.FIRST_PLAYER
        board[y - 1][x] = CellState.FIRST_PLAYER
        board[y - 1][x - 1] = CellState.SECOND_PLAYER
        return OthelloState(Player.FIRST, tuple(tuple(row) for row in board))

    @lru_cache(maxsize=2 ** 20)
    def generate_child_state(  # type: ignore[override]
        self, state: OthelloState, action: Action
    ) -> OthelloState:
        assert action in self.legal_actions(state)
        if action == self.pass_move:
            return dataclasses.replace(state, player=state.player.opposite())
        x, y = action % self.grid_size, action // self.grid_size
        grid = [[cell for cell in row] for row in state.grid]
        grid[y][x] = CellState(state.player)
        opposite_player = state.player.opposite()
        shifts = itertools.product([0, 1, -1], repeat=2)
        for x_shift, y_shift in shifts:
            shifted_x = x
            shifted_y = y
            opposites = []
            while True:
                shifted_x += x_shift
                shifted_y += y_shift
                if not (0 <= shifted_x < len(grid)) or not (0 <= shifted_y < len(grid)):
                    break
                if grid[shifted_y][shifted_x] == CellState(opposite_player):
                    opposites.append((shifted_x, shifted_y))
                else:
                    break
            if (
                (0 <= shifted_x < len(grid))
                and (0 <= shifted_y < len(grid))
                and grid[shifted_y][shifted_x] == CellState(state.player)
            ):
                for opposite_x, opposite_y in opposites:
                    grid[opposite_y][opposite_x] = CellState(state.player)
        return OthelloState(opposite_player, tuple(tuple(row) for row in grid))

    @lru_cache(maxsize=2 ** 20)
    def legal_actions(  # type: ignore[override]
        self, state: OthelloState
    ) -> List[Action]:
        actions: Set[Action] = set()
        opposite_player = state.player.opposite()
        for x, y in player_positions(state):
            shifts = itertools.product([0, 1, -1], repeat=2)
            for x_shift, y_shift in shifts:
                shifted_x = x
                shifted_y = y
                found_opposite = False
                while True:
                    shifted_x += x_shift
                    shifted_y += y_shift
                    if not (0 <= shifted_x < self.grid_size) or not (
                        0 <= shifted_y < self.grid_size
                    ):
                        break
                    if state.grid[shifted_y][shifted_x] == CellState(opposite_player):
                        found_opposite = True
                    else:
                        break
                if (
                    found_opposite
                    and (0 <= shifted_x < self.grid_size)
                    and (0 <= shifted_y < self.grid_size)
                    and state.grid[shifted_y][shifted_x] == CellState.EMPTY
                ):
                    action = shifted_y * self.grid_size + shifted_x
                    assert action < self.num_actions
                    actions.add(action)
        if not actions:
            actions.add(self.pass_move)
        return list(actions)

    @lru_cache(maxsize=2 ** 20)
    def is_final_state(self, state: OthelloState) -> bool:  # type: ignore[override]
        return self.legal_actions(state) == [self.pass_move] and self.legal_actions(
            self.generate_child_state(state, self.pass_move)
        ) == [self.pass_move]

    @lru_cache(maxsize=2 ** 20)
    def evaluate_final_state(  # type: ignore[override]
        self, state: OthelloState
    ) -> Outcome:
        first_player_pieces = sum(
            sum(c == CellState.FIRST_PLAYER for c in row) for row in state.grid
        )
        second_player_pieces = sum(
            sum(c == CellState.SECOND_PLAYER for c in row) for row in state.grid
        )
        if first_player_pieces > second_player_pieces:
            return Outcome.FIRST_PLAYER_WIN
        elif second_player_pieces > first_player_pieces:
            return Outcome.SECOND_PLAYER_WIN
        else:
            return Outcome.DRAW

    def probabilities_grid(self, action_probabilities: Sequence[float]) -> str:
        board = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for action, probability in enumerate(action_probabilities[:-1]):
            x, y = action % self.grid_size, action // self.grid_size
            board[y][x] = probability
        grid = []
        for i, row in enumerate(board, 1):
            row_str = " ".join(f"{x:.2f}" for x in row)
            grid.append(row_str)
        grid.append(f"pass: {action_probabilities[self.pass_move]}")
        return "\n".join(grid)

    def action_str(self, action: Action) -> str:
        if action == self.pass_move:
            return "pass"
        return super().action_str(action)


def player_positions(state: OthelloState) -> Iterator[Tuple[int, int]]:
    for y, row in enumerate(state.grid):
        for x, cell in enumerate(row):
            if cell == CellState(state.player):
                yield x, y


if __name__ == "__main__":
    play_random_mcts(OthelloManager(grid_size=6), num_simulations=100)
