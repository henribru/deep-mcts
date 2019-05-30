from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

from deep_mcts.mcts import State, Action, GameManager, MCTS


@dataclass(frozen=True)
class HexState(State):
    grid: List[List[int]]


@dataclass(frozen=True)
class HexAction(Action):
    coordinate: Tuple[int, int]


class HexManager(GameManager[HexState, HexAction]):
    grid_size: int

    def __init__(self, grid_size: int):
        super().__init__()
        self.grid_size = grid_size

    def initial_game_state(self) -> HexState:
        return HexState(
            0, [[-1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        )

    def generate_child_states(self, parent: HexState) -> Dict[HexAction, HexState]:
        child_states = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if parent.grid[y][x] == -1:
                    child_states[HexAction((x, y))] = HexState(
                        (parent.player + 1) % 2,
                        [
                            [
                                parent.player if (i, j) == (x, y) else parent.grid[j][i]
                                for i in range(self.grid_size)
                            ] if j == y else parent.grid[j]
                            for j in range(self.grid_size)
                        ],
                    )
        return child_states

    def generate_child_state(self, parent: HexState, action: HexAction) -> HexState:
        return HexState(
            (parent.player + 1) % 2,
            [
                [
                    parent.player if (i, j) == action.coordinate else parent.grid[j][i]
                    for i in range(self.grid_size)
                ] if j == action.coordinate[1] else parent.grid[j]
                for j in range(self.grid_size)
            ],
        )

    def legal_actions(self, state: HexState) -> List[HexAction]:
        actions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if state.grid[y][x] == -1:
                    actions.append(HexAction((x, y)))
        return actions

    def is_final_state(self, state: HexState) -> bool:
        return self.evaluate_final_state(state) != 0

    def evaluate_final_state(self, state: HexState) -> float:
        starts = ((0, y) for y in range(self.grid_size) if state.grid[y][0] == 0)
        if any(self.traverse_from(start, 0, state) for start in starts):
            return 1
        starts = ((x, 0) for x in range(self.grid_size) if state.grid[0][x] == 1)
        if any(self.traverse_from(start, 1, state) for start in starts):
            return -1
        return 0

    def traverse_from(
        self, coordinate: Tuple[int, int], player: int, state: HexState, visited=None
    ):
        if visited is None:
            visited = []
        if coordinate in visited:
            return False
        if (
            player == 0
            and coordinate[0] == self.grid_size - 1
            or player == 1
            and coordinate[1] == self.grid_size - 1
        ):
            return True
        visited.append(coordinate)
        return any(
            self.traverse_from(neighbour, player, state, visited)
            for neighbour in self.get_neighbours(coordinate, player, state)
        )

    def get_neighbours(self, coordinate: Tuple[int, int], player: int, state: HexState):
        x, y = coordinate
        shifts = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]
        for y_shift, x_shift in shifts:
            shifted_x = x + x_shift
            shifted_y = y + y_shift
            if (
                0 <= shifted_x < self.grid_size
                and 0 <= shifted_y < self.grid_size
                and state.grid[shifted_y][shifted_x] == player
            ):
                yield shifted_x, shifted_y


def hex_simulator(grid_size: int, M: int):
    def state_evaluator(state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        legal_actions = hex.legal_actions(state)
        return 0, {action: 1 / len(legal_actions) for action in hex.legal_actions(state)}
    hex = HexManager(grid_size)
    mcts = MCTS(hex, M, lambda state: random.choice(hex.legal_actions(state)), state_evaluator)
    for state, next_state, action, visit_distribution in mcts.run():
        print(action.coordinate)
        print_hex_grid(state.grid)
        print("-" * 5)
    print_hex_grid(next_state.grid)
    print(state.player)


def print_hex_grid(grid):
    symbol = {-1: "#", 0: "0", 1: "1"}
    for i in range(len(grid) - 1):
        print(
            format(
                " ".join(symbol[grid[i - x][x]] for x in range(i + 1)),
                f"^{2 * len(grid) - 1}",
            )
        )
    for i in range(len(grid)):
        print(
            format(
                " ".join(
                    symbol[grid[len(grid) - 1 + i - x][x]] for x in range(i, len(grid))
                ),
                f"^{2 * len(grid) - 1}",
            )
        )


if __name__ == "__main__":
    hex_simulator(4, 100)
