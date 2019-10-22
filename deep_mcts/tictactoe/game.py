import random
from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass

from deep_mcts.game import GameManager, State, Player, CellState
from deep_mcts.mcts import MCTS


@dataclass(frozen=True)
class TicTacToeState(State):
    grid: List[List[CellState]]


@dataclass(frozen=True)
class TicTacToeAction:
    coordinate: Tuple[int, int]


class TicTacToeManager(GameManager[TicTacToeState, TicTacToeAction]):
    def initial_game_state(self) -> TicTacToeState:
        return TicTacToeState(
            Player.FIRST, [[CellState.EMPTY for _ in range(3)] for _ in range(3)]
        )

    def generate_child_states(
        self, state: TicTacToeState
    ) -> Dict[TicTacToeAction, TicTacToeState]:
        child_states = {
            action: self.generate_child_state(state, action)
            for action in self.legal_actions(state)
        }
        assert set(child_states.keys()) == set(self.legal_actions(state))
        return child_states

    def generate_child_state(
        self, state: TicTacToeState, action: TicTacToeAction
    ) -> TicTacToeState:
        assert action in self.legal_actions(state)
        x, y = action.coordinate
        return TicTacToeState(
            state.player.opposite(),
            [
                [
                    CellState(state.player) if (i, j) == (x, y) else state.grid[j][i]
                    for i in range(3)
                ]
                if j == y
                else state.grid[j]
                for j in range(3)
            ],
        )

    def legal_actions(self, state: TicTacToeState) -> List[TicTacToeAction]:
        return [
            TicTacToeAction((x, y))
            for y in range(3)
            for x in range(3)
            if state.grid[y][x] == CellState.EMPTY
        ]

    def is_final_state(self, state: TicTacToeState) -> bool:
        return self.evaluate_final_state(state) != 0 or all(
            all(p != CellState.EMPTY for p in row) for row in state.grid
        )

    def evaluate_final_state(self, state: TicTacToeState) -> int:
        for player, outcome in [
            (CellState.SECOND_PLAYER, 1),
            (CellState.FIRST_PLAYER, -1),
        ]:
            if (
                any(all(p == player for p in state.grid[y]) for y in range(3))
                or any(
                    all(state.grid[y][x] == player for y in range(3)) for x in range(3)
                )
                or all(state.grid[i][i] == player for i in range(3))
                or all(state.grid[i][2 - i] == player for i in range(3))
            ):
                return outcome
        return 0


def tic_tac_toe_simulator(num_simulations: int) -> None:
    manager = TicTacToeManager()
    mcts = MCTS(
        manager,
        num_simulations,
        lambda state: random.choice(manager.legal_actions(state)),
        None,
    )
    for state, next_state, action, _ in mcts.self_play():
        print(action.coordinate)
        print_tic_tac_toe_grid(state.grid)
        print("-" * 5)
    print_tic_tac_toe_grid(next_state.grid)
    print(manager.evaluate_final_state(next_state))


def print_tic_tac_toe_grid(grid: Iterable[Iterable[int]]) -> None:
    symbol = {-1: "#", 0: "X", 1: "O"}
    for row in grid:
        for p in row:
            print(symbol[p], end="")
        print()


if __name__ == "__main__":
    tic_tac_toe_simulator(1000)

r"""C:\Users\henbruas\AppData\Local\pypoetry\Cache\virtualenvs\deep-mcts-py3.7\Scripts\python.exe "D:/OneDrive - NTNU/NTNU/IT3105/Deep MCTS/deep_mcts/tictactoe/game.py"
(0, 2)
###
###
###
-----
(0, 1)
###
###
X##
-----
(1, 1)
###
O##
X##
-----
(2, 0)
###
OX#
X##
-----
(2, 2)
##O
OX#
X##
-----
(1, 2)
##O
OX#
X#X
-----
(0, 0)
##O
OX#
XOX
-----
X#O
OX#
XOX
0

Process finished with exit code 0
"""
