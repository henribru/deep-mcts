from collections import deque

from anet import ANET
from convolutional_hex import ConvolutionalHexANET
from mcts import MCTS, StateManager
from topp import topp
import functools


def train(
    anet: ANET,
    state_manager: StateManager,
    num_actual_games: int,
    num_search_games: int,
    save_interval: int,
):
    replay_buffer = deque([], 100000)
    random_anet = ConvolutionalHexANET(grid_size=4)
    anet.save(f"anet-0.pth")
    for i in range(num_actual_games):
        print(i + 1)
        mcts = MCTS(state_manager, num_search_games, functools.partial(anet.greedy_policy, epsilon=0.10), anet.evaluate_state)
        examples = []
        for state, next_state, action, visit_distribution in mcts.run():
            examples.append((state, visit_distribution))
            yield state, next_state, action, visit_distribution
        value = 1 if state.player == 0 else -1
        for example in examples:
            state, visit_distribution = example
            replay_buffer.append((state, visit_distribution, value))
        anet.train(replay_buffer)
        results = topp([functools.partial(random_anet.greedy_policy, epsilon=0.10), functools.partial(anet.greedy_policy, epsilon=0.10)], 25, state_manager)
        print(results)
        if (i + 1) % save_interval == 0:
            anet.save(f"anet-{i + 1}.pth")
