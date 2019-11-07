from typing import Dict

from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.tictactoe.fullyconnectednet import FullyConnectedTicTacToeNet
from deep_mcts.tictactoe.game import TicTacToeAction, TicTacToeManager


def print_probabilities_grid(probabilities: Dict[TicTacToeAction, float]) -> None:
    grid = [[0.0 for _ in range(3)] for _ in range(3)]
    for action, probability in probabilities.items():
        x, y = action.coordinate
        grid[y][x] = probability
    for row in grid:
        for probability in row:
            print(f"{probability:.5f}", end="\t")
        print()


if __name__ == "__main__":
    net = FullyConnectedTicTacToeNet.from_path("saves/anet-1000.pth")
    manager = TicTacToeManager()
    mcts = MCTS(manager, 100, None, net.evaluate_state)
    agent = MCTSAgent(mcts)
    # print(tournament([agent, RandomAgent(manager)], 100, manager))
    # for state, next_state, action, visit_distribution in mcts.self_play():
    #     print_tic_tac_toe_grid(state.grid)
    #     print()
    #     print_probabilities_grid(visit_distribution)
    #     print()
    #     print_probabilities_grid(net.evaluate_state(state)[1])
    #     print("---")
    # print_tic_tac_toe_grid(next_state.grid)
    # print(manager.evaluate_final_state(next_state))
    # net = FullyConnectedTicTacToeNet.from_path("saves/anet-65500.pth")
    # grid = [[0, 0, -1], [-1, -1, 1], [1, -1, -1]]
    # state = TicTacToeState(0, grid)
    # mcts.root = Node(state)
    # print_probabilities_grid(mcts.step())
    # print(net._distributions_to_tensor([state], [{TicTacToeAction((2, 0)): 1, TicTacToeAction((0, 1)): 0, TicTacToeAction((1, 1)): 0, TicTacToeAction((1, 2)): 0, TicTacToeAction((2, 2)): 0}]))
    # value, probabilities = net.evaluate_state(state)
    # print(value)
    # print_probabilities_grid(probabilities)
    # # for _ in range(1000):
    # net.train([(state, {TicTacToeAction((2, 0)): 1, TicTacToeAction((0, 1)): 0, TicTacToeAction((1, 1)): 0, TicTacToeAction((1, 2)): 0, TicTacToeAction((2, 2)): 0}, 1)])
    # print_tic_tac_toe_grid(grid)
    # value, probabilities = net.evaluate_state(state)
    # print(value)
    # print_probabilities_grid(probabilities)
