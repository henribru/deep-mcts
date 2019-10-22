import random
import string
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, MutableSet, Optional, Set

from deep_mcts.mcts import MCTS
from deep_mcts.game import GameManager, Player, State, CellState


@dataclass(frozen=True)
class HexState(State):
    grid: List[List[CellState]]

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


@dataclass(frozen=True)
class HexAction:
    coordinate: Tuple[int, int]


class HexManager(GameManager[HexState, HexAction]):
    grid_size: int

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size

    def initial_game_state(self) -> HexState:
        return HexState(
            Player.FIRST,
            [
                [CellState.EMPTY for _ in range(self.grid_size)]
                for _ in range(self.grid_size)
            ],
        )

    def generate_child_states(self, state: HexState) -> Dict[HexAction, HexState]:
        child_states = {
            action: self.generate_child_state(state, action)
            for action in self.legal_actions(state)
        }
        assert set(child_states.keys()) == set(self.legal_actions(state))
        return child_states

    def generate_child_state(self, state: HexState, action: HexAction) -> HexState:
        assert action in self.legal_actions(state)
        x, y = action.coordinate
        return HexState(
            state.player.opposite(),
            [
                [
                    CellState(state.player) if (i, j) == (x, y) else state.grid[j][i]
                    for i in range(self.grid_size)
                ]
                if j == y
                else state.grid[j]
                for j in range(self.grid_size)
            ],
        )

    def legal_actions(self, state: HexState) -> List[HexAction]:
        return [
            HexAction((x, y))
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if state.grid[y][x] == CellState.EMPTY
        ]

    def is_final_state(self, state: HexState) -> bool:
        return self.evaluate_final_state(state) != 0

    def evaluate_final_state(self, state: HexState) -> int:
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
            return 1
        starts = (
            (x, 0)
            for x in range(self.grid_size)
            if state.grid[0][x] == CellState.FIRST_PLAYER
        )
        if any(
            self._traverse_from(start, Player.FIRST, state, visited) for start in starts
        ):
            return -1
        return 0

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
        self, coordinate: Tuple[int, int], player: int, state: HexState
    ) -> Iterable[Tuple[int, int]]:
        x, y = coordinate
        shifts = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
        for x_shift, y_shift in shifts:
            shifted_x = x + x_shift
            shifted_y = y + y_shift
            if (
                0 <= shifted_x < self.grid_size
                and 0 <= shifted_y < self.grid_size
                and state.grid[shifted_y][shifted_x] == player
            ):
                yield shifted_x, shifted_y


def hex_simulator(grid_size: int, num_simulations: int) -> None:
    def state_evaluator(state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        legal_actions = manager.legal_actions(state)
        return 0, {action: 1 / len(legal_actions) for action in legal_actions}

    manager = HexManager(grid_size)
    mcts = MCTS(
        manager,
        num_simulations,
        lambda state: random.choice(manager.legal_actions(state)),
        state_evaluator,
    )
    for state, next_state, action, _ in mcts.self_play():
        print(action.coordinate)
        print(state)
        print("-" * 5)
    print(next_state)
    print(state.player)


if __name__ == "__main__":
    hex_simulator(4, 1000)
