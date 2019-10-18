import queue
import random
import time
from typing import Iterable, Tuple, Optional, TypeVar, Callable, Deque, Dict, List

import torch
from torch import multiprocessing

from deep_mcts.game import State, Action, GameManager
from deep_mcts.gamenet import GameNet, DEVICE
from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.tournament import RandomAgent, compare_agents

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


def create_self_play_examples(
    process_number: int,
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    rollout_policy: Optional[Callable[[_S], _A]],
    games_queue: "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]",
    epsilon: float,
) -> None:
    game_net.net.eval()
    for i in range(num_games):
        mcts = MCTS(
            game_manager,
            num_simulations,
            rollout_policy,
            state_evaluator=game_net.evaluate_state,
            epsilon=epsilon,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.self_play():
            examples.append((state, visit_distribution))
        outcome = game_manager.evaluate_final_state(next_state)
        games_queue.put(
            [
                (state, visit_distribution, outcome if state.player == 0 else -outcome)
                for state, visit_distribution in examples
            ]
        )
        if i % 100 == 0:
            print(process_number, i)


def train(
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    save_interval: int,
    evaluation_interval: int,
    rollout_policy: Optional[Callable[[_S], _A]] = None,
) -> Iterable[
    Tuple[int, Tuple[float, float, float], Optional[Tuple[float, float, float]]]
]:
    replay_buffer = Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]([], 100_000)
    game_net.save(f"saves/anet-0.pth")
    random_opponent = RandomAgent(game_manager)
    original_opponent = game_net.copy()
    epsilon = 0.05
    original_agent = MCTSAgent(
        MCTS(
            game_manager,
            num_simulations,
            rollout_policy,
            state_evaluator=original_opponent.evaluate_state,
            epsilon=epsilon,
        )
    )
    previous_agent: Optional[MCTSAgent[_S, _A]] = None
    multiprocessing.set_start_method("spawn")
    games_queue: "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]" = multiprocessing.Queue()
    spawn_context = multiprocessing.spawn(
        create_self_play_examples,
        (
            game_net,
            game_manager,
            num_games,
            num_simulations,
            rollout_policy,
            games_queue,
            epsilon,
        ),
        nprocs=26,
        join=False,
    )
    i = 0
    prev_evaluation_time = time.perf_counter()
    start_time = time.perf_counter()
    while not spawn_context.join(0):
        new_examples: List[Tuple[_S, Dict[_A, float], float]] = []
        while True:
            try:
                block = not replay_buffer and not new_examples
                new_examples.extend(games_queue.get(block, timeout=1))
            except queue.Empty:
                if block and not spawn_context.join(0):
                    continue
                break
        if new_examples:
            states, probability_targets, value_targets = zip(*new_examples)
            value_targets = torch.tensor(value_targets, dtype=torch.float32).reshape(
                (-1, 1)
            )
            assert value_targets.shape[0] == len(new_examples)
            probability_targets = game_net.distributions_to_tensor(
                states, probability_targets
            )
            assert probability_targets.shape[0] == len(new_examples)
            states = game_net.states_to_tensor(states)
            assert states.shape[0] == len(new_examples)
            for j in range(len(new_examples)):
                replay_buffer.append(
                    (states[j], probability_targets[j], value_targets[j])
                )
            # print(len(replay_buffer) / (time.perf_counter() - start_time))
        examples = random.sample(replay_buffer, min(512, len(replay_buffer)))
        states, probability_targets, value_targets = zip(*examples)  # type: ignore
        states = torch.stack(states).to(DEVICE)  # type: ignore
        probability_targets = torch.stack(probability_targets).to(  # type: ignore
            DEVICE
        )
        value_targets = torch.stack(value_targets).to(DEVICE)  # type: ignore
        game_net.train(states, probability_targets, value_targets)
        if evaluation_interval != 0 and (i + 1) % evaluation_interval == 0:
            game_net.net.eval()
            print("evaluating")
            best_agent = MCTSAgent(
                MCTS(
                    game_manager,
                    100,
                    rollout_policy,
                    state_evaluator=game_net.evaluate_state,
                )
            )
            sampling_agent = MCTSAgent(
                MCTS(
                    game_manager,
                    100,
                    rollout_policy,
                    state_evaluator=game_net.evaluate_state,
                    epsilon=epsilon,
                )
            )
            random_evaluation = compare_agents(
                (best_agent, random_opponent), 20, game_manager
            )
            random_mcts_evaluation = compare_agents(
                (
                    best_agent,
                    MCTSAgent(
                        MCTS(
                            game_manager,
                            num_simulations,
                            lambda s: random.choice(game_manager.legal_actions(s)),
                            state_evaluator=None,
                        )
                    ),
                ),
                20,
                game_manager,
            )
            original_evaluation = compare_agents(
                (sampling_agent, original_agent), 20, game_manager
            )
            previous_evaluation = (
                compare_agents((sampling_agent, previous_agent), 20, game_manager)
                if previous_agent is not None
                else original_evaluation
            )
            previous_agent = MCTSAgent(
                MCTS(
                    game_manager,
                    num_simulations,
                    rollout_policy,
                    state_evaluator=game_net.copy().evaluate_state,
                    epsilon=epsilon,
                )
            )
            print(
                f"i: {i + 1}, t: {time.perf_counter() - prev_evaluation_time} random: {random_evaluation} "
                f"previous: {previous_evaluation} original: {original_evaluation} random MCTS: {random_mcts_evaluation} "
                f"moves: {len(replay_buffer)}"
            )
            prev_evaluation_time = time.perf_counter()
            game_net.net.train()
            yield i + 1, random_evaluation, previous_evaluation
        if save_interval != 0 and (i + 1) % save_interval == 0:
            filepath = f"saves/anet-{i + 1}.pth"
            game_net.save(filepath)
            print(f"Saved {filepath}")
        i += 1
