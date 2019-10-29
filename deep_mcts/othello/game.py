from functools import lru_cache

import dataclasses
import itertools
import random
import string
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union, Set, Mapping

from deep_mcts.game import CellState, GameManager, Outcome, Player, State
from deep_mcts.mcts import MCTS


@dataclass(frozen=True)
class OthelloState(State):
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


@dataclass(frozen=True)
class OthelloMove:
    coordinate: Tuple[int, int]

    def generate_child_state(self, state: OthelloState) -> OthelloState:
        x, y = self.coordinate
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


@dataclass(frozen=True)
class OthelloPass:
    def generate_child_state(self, state: OthelloState) -> OthelloState:
        return dataclasses.replace(state, player=state.player.opposite())


OthelloAction = Union[OthelloMove, OthelloPass]


class OthelloManager(GameManager[OthelloState, OthelloAction]):
    grid_size: int

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size

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
        self, state: OthelloState, action: OthelloAction
    ) -> OthelloState:
        assert action in self.legal_actions(state)
        return action.generate_child_state(state)

    @lru_cache(maxsize=2 ** 20)
    def legal_actions(  # type: ignore[override]
        self, state: OthelloState
    ) -> List[OthelloAction]:
        actions: Set[OthelloAction] = set()
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
                    actions.add(OthelloMove((shifted_x, shifted_y)))
        if not actions:
            actions.add(OthelloPass())
        return list(actions)

    @lru_cache(maxsize=2 ** 20)
    def is_final_state(self, state: OthelloState) -> bool:  # type: ignore[override]
        return self.legal_actions(state) == [OthelloPass()] and self.legal_actions(
            self.generate_child_state(state, OthelloPass())
        ) == [OthelloPass()]

    @lru_cache(maxsize=2 ** 20)
    def evaluate_final_state(  # type: ignore[override]
        self, state: OthelloState
    ) -> int:
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


def player_positions(state: OthelloState) -> Iterator[Tuple[int, int]]:
    for y, row in enumerate(state.grid):
        for x, cell in enumerate(row):
            if cell == CellState(state.player):
                yield x, y


def othello_simulator(grid_size: int, num_simulations: int) -> None:
    manager = OthelloManager(grid_size)
    mcts = MCTS(
        manager,
        num_simulations,
        lambda state: random.choice(manager.legal_actions(state)),
        None,
    )
    for state, next_state, action, _ in mcts.self_play():
        print(state)
        print("-" * 5)
        print(getattr(action, "coordinate", "pass"))
    print(next_state)
    print(manager.evaluate_final_state(next_state))


def othello_probabilities_grid(
    action_probabilities: Mapping[OthelloAction, float], grid_size: int
) -> str:
    board = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for action, probability in action_probabilities.items():
        if not isinstance(action, OthelloPass):
            x, y = action.coordinate
            board[y][x] = probability
    grid = []
    for i, row in enumerate(board, 1):
        row_str = " ".join(f"{x:.2f}" for x in row)
        grid.append(row_str)
    pass_ = OthelloPass()
    if pass_ in action_probabilities:
        grid.append(f"pass: {action_probabilities[pass_]}")
    return "\n".join(grid)


if __name__ == "__main__":
    othello_simulator(6, 100)
