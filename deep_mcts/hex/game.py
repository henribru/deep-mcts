import string
from functools import lru_cache
from typing import Tuple, List, Iterable, MutableSet, Optional, Set, Sequence

from dataclasses import dataclass

from deep_mcts.game import GameManager, Player, State, CellState, Outcome, Action
from deep_mcts.mcts import play_random_mcts


@dataclass(unsafe_hash=True)
class HexState(State):
    __slots__ = ["grid"]
    grid: Tuple[Tuple[CellState, ...], ...]

    def __str__(self) -> str:
        symbol = {-1: ".", 0: "0", 1: "1"}
        grid = []
        letters = string.ascii_uppercase[: len(self.grid)]
        width = 2 * len(self.grid) - 1 + 4
        grid.append(" ".join(letters).center(width - 1))
        for i in range(len(self.grid)):
            tiles = " ".join(symbol[x] for x in self.grid[i])
            grid.append(
                f"{' ' * (i if i < 9 else i - 1)}{i + 1} {tiles} {i + 1}".center(width)
            )
        grid.append(f"{' ' * (len(self.grid) + 2)}{' '.join(letters)}".center(width))
        return "\n".join(grid)


class HexManager(GameManager[HexState]):
    def __init__(self, grid_size: int, num_actions: Optional[int] = None) -> None:
        if num_actions is None:
            num_actions = grid_size ** 2
        super().__init__(grid_size, num_actions)

    def initial_game_state(self) -> HexState:
        return HexState(
            Player.FIRST,
            tuple(
                tuple(CellState.EMPTY for _ in range(self.grid_size))
                for _ in range(self.grid_size)
            ),
        )

    @lru_cache(maxsize=2 ** 20)
    def generate_child_state(  # type: ignore[override]
        self, state: HexState, action: Action
    ) -> HexState:
        assert action in self.legal_actions(state)
        x, y = action % self.grid_size, action // self.grid_size
        return HexState(
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
        self, state: HexState
    ) -> List[Action]:
        return [
            x + y * self.grid_size
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if state.grid[y][x] == CellState.EMPTY
        ]

    @lru_cache(maxsize=2 ** 20)
    def is_final_state(  # type: ignore[override]
        self, state: HexState
    ) -> bool:
        return self.evaluate_final_state(state) != Outcome.DRAW

    @lru_cache(maxsize=2 ** 20)
    def evaluate_final_state(self, state: HexState) -> Outcome:  # type: ignore[override]
        starts = (
            (0, y)
            for y in range(self.grid_size)
            if state.grid[y][0] == CellState.SECOND_PLAYER
        )
        visited: Set[Tuple[int, int]] = set()
        if any(
            self._traverse_from(start, Player.SECOND, state, visited)
            for start in starts
        ):
            return Outcome.SECOND_PLAYER_WIN
        starts = (
            (x, 0)
            for x in range(self.grid_size)
            if state.grid[0][x] == CellState.FIRST_PLAYER
        )
        if any(
            self._traverse_from(start, Player.FIRST, state, visited) for start in starts
        ):
            return Outcome.FIRST_PLAYER_WIN
        return Outcome.DRAW

    def _traverse_from(
        self,
        coordinate: Tuple[int, int],
        player: Player,
        state: HexState,
        visited: Optional[MutableSet[Tuple[int, int]]] = None,
    ) -> bool:
        if visited is None:
            visited = set()
        if coordinate in visited:
            return False
        visited.add(coordinate)
        x, y = coordinate
        if (
            player == Player.SECOND
            and x == self.grid_size - 1
            or player == Player.FIRST
            and y == self.grid_size - 1
        ):
            return True
        return any(
            self._traverse_from(neighbour, player, state, visited)
            for neighbour in self._get_neighbours(coordinate, player, state)
        )

    def _get_neighbours(
        self, coordinate: Tuple[int, int], player: Player, state: HexState
    ) -> Iterable[Tuple[int, int]]:
        x, y = coordinate
        shifts = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
        for x_shift, y_shift in shifts:
            shifted_x = x + x_shift
            shifted_y = y + y_shift
            if (
                0 <= shifted_x < self.grid_size
                and 0 <= shifted_y < self.grid_size
                and state.grid[shifted_y][shifted_x] == CellState(player)
            ):
                yield shifted_x, shifted_y

    def probabilities_grid(self, action_probabilities: Sequence[float]) -> str:
        board = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for action, probability in enumerate(action_probabilities):
            x, y = action % self.grid_size, action // self.grid_size
            board[y][x] = probability
        grid = []
        width = 4 * len(board) - 1 + 4
        for i in range(len(board)):
            tiles = " ".join(f"{x:.2f}" for x in board[i])
            grid.append(
                f"{' ' * (i if i < 9 else i - 1)}{i + 1} {tiles} {i + 1}".center(width)
            )
        return "\n".join(grid)


if __name__ == "__main__":
    play_random_mcts(HexManager(grid_size=4), num_simulations=1000)
