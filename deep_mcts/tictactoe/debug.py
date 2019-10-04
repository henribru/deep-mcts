from typing import Dict

from deep_mcts.tictactoe.convolutionalnet import ConvolutionalTicTacToeNet
from deep_mcts.tictactoe.fullyconnectednet import FullyConnectedTicTacToeNet
from deep_mcts.tictactoe.game import TicTacToeState, TicTacToeAction, print_tic_tac_toe_grid


def print_probabilities_grid(probabilities: Dict[TicTacToeAction, float]) -> None:
    grid = [[0 for _ in range(3)] for _ in range(3)]
    for action, probability in probabilities.items():
        x, y = action.coordinate
        grid[y][x] = probability
    for row in grid:
        for probability in row:
            print(f"{probability:.2f}", end="\t")
        print()


if __name__ == "__main__":
    # net = ConvolutionalTicTacToeNet()
    # net.save("debug.pth")
    net = FullyConnectedTicTacToeNet.from_path("anet-800.pth")
    grid = [[0, 0, -1], [-1, -1, 1], [1, -1, -1]]
    print_tic_tac_toe_grid(grid)
    value, probabilities = net.evaluate_state(TicTacToeState(1, grid))
    print(value)
    print_probabilities_grid(probabilities)
