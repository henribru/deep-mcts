import queue
import random
import time
from functools import lru_cache
from typing import Iterable, Tuple, Optional, TypeVar, Callable, Dict, List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import multiprocessing

from deep_mcts.game import State, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS, MCTSAgent, Player
from deep_mcts.tournament import compare_agents, AgentComparison

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")
SelfPlayExample = Tuple[_S, Dict[_A, float], float]
SelfPlayGame = List[SelfPlayExample[_S, _A]]
TensorSelfPlayExample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TensorSelfPlayGame = List[TensorSelfPlayExample]

# def train_from_checkpoint(anet: GameNet[_S, _A], path: str) -> None:
#     save_dir = Path(path).resolve().parent
#     checkpoint = torch.load(path)
#     anet.load_state_dict(checkpoint["model_state_dict"])
#     replay_buffer = checkpoint["replay_buffer"]
#     anet.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     training_iterations = checkpoint["training_iterations"]


@dataclass(frozen=True)
class TrainingConfiguration:
    num_games: int
    num_simulations: int
    save_interval: int
    evaluation_interval: int
    save_dir: str
    sample_move_cutoff: int
    dirichlet_alpha: float
    dirichlet_factor: float = 0.25
    rollout_policy: Optional[Callable[[_S], _A]] = None
    epsilon: float = 0.05
    nprocs: int = 25
    batch_size: int = 512
    replay_buffer_max_size: int = 100_000
    train_device: torch.device = torch.device("cuda:1")
    self_play_device: torch.device = torch.device("cuda:0")
    evaluation_games: int = 20
    transfer_interval: int = 1000


def train(game_net: GameNet[_S, _A], config: TrainingConfiguration,) -> None:
    evaluations = pd.DataFrame.from_dict(
        {
            i: (random_evaluation, previous_evaluation)
            for i, random_evaluation, previous_evaluation in _train(game_net, config,)
        },
        orient="index",
        columns=["against_random", "against_previous"],
    )
    evaluations.to_csv(f"{config.save_dir}/evaluations.csv")


def _train(
    game_net: GameNet[_S, _A], config: TrainingConfiguration,
) -> Iterable[Tuple[int, AgentComparison, AgentComparison]]:
    print(f"{time.strftime('%H:%M:%S')} Starting")
    game_manager = game_net.manager
    game_net.to(config.train_device)
    replay_buffer: List[TensorSelfPlayGame] = []
    game_net.save(f"{config.save_dir}/anet-0.pth")
    multiprocessing.set_start_method("spawn")
    self_play_game_net = game_net.copy().to(config.self_play_device)
    self_playing_context, games_queue = spawn_self_play_example_creators(
        self_play_game_net, config,
    )
    previous_game_net = game_net.copy().to(config.train_device)
    training_iterations = 0
    training_games_count = 0
    training_examples_count = 0
    prev_evaluation_time = time.perf_counter()
    while not self_playing_context.join(0):
        new_games = get_new_games(
            games_queue, self_playing_context, game_net, block=len(replay_buffer) == 0
        )
        training_games_count += len(new_games)
        training_examples_count += sum(len(game) for game in new_games)
        replay_buffer.extend(new_games)
        if len(replay_buffer) > config.replay_buffer_max_size:
            replay_buffer = replay_buffer[-config.replay_buffer_max_size :]
        states, probability_targets, value_targets = sample_replay_buffer(
            replay_buffer, config.batch_size, config.train_device
        )
        game_net.train(states, probability_targets, value_targets)

        if (training_iterations + 1) % config.transfer_interval == 0:
            self_play_game_net.load_state_dict(game_net.net.state_dict())

        if (
            config.save_interval != 0
            and (training_iterations + 1) % config.save_interval == 0
        ):
            filepath = f"{config.save_dir}/anet-{training_iterations + 1}.pth"
            game_net.save(filepath)
            print(f"{time.strftime('%H:%M:%S')} Saved {filepath}")

        if (
            config.evaluation_interval != 0
            and (training_iterations + 1) % config.evaluation_interval == 0
        ):
            print(f"{time.strftime('%H:%M:%S')} evaluating")
            random_mcts_evaluation, previous_evaluation = evaluate(
                game_net, previous_game_net, game_manager, config,
            )
            print(
                f"{time.strftime('%H:%M:%S')} "
                f"iterations: {training_iterations} games: {training_games_count} "
                f"examples: {training_examples_count} evaluation_duration: {time.perf_counter() - prev_evaluation_time:.2f} "
                f"previous: {previous_evaluation} random MCTS: {random_mcts_evaluation} "
            )
            prev_evaluation_time = time.perf_counter()
            previous_game_net.load_state_dict(game_net.net.state_dict())
            yield training_iterations, random_mcts_evaluation, previous_evaluation

        training_iterations += 1


def spawn_self_play_example_creators(
    game_net: GameNet[_S, _A], config: TrainingConfiguration,
) -> Tuple[multiprocessing.SpawnContext, "multiprocessing.Queue[SelfPlayGame[_S, _A]]"]:
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S, _A]]" = multiprocessing.Queue()
    context = multiprocessing.spawn(
        create_self_play_examples,
        (game_net, config, games_queue),
        nprocs=config.nprocs,
        join=False,
    )
    return context, games_queue


def create_self_play_examples(
    process_number: int,
    game_net: GameNet[_S, _A],
    config: TrainingConfiguration,
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S, _A]]",
) -> None:
    game_manager = game_net.manager
    for i in range(config.num_games):
        mcts = MCTS(
            game_manager,
            config.num_simulations,
            config.rollout_policy,
            cached_state_evaluator(game_net),
            config.sample_move_cutoff,
            config.dirichlet_alpha,
            config.dirichlet_factor,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.self_play():
            examples.append((state, visit_distribution))
        outcome = game_manager.evaluate_final_state(next_state)
        games_queue.put(
            [
                (
                    state,
                    visit_distribution,
                    outcome if state.player == Player.SECOND else -outcome,
                )
                for state, visit_distribution in examples
            ]
        )
        if i % 100 == 0 and process_number == 0:
            print(f"{time.strftime('%H:%M:%S')} {i}")
            cached_methods = [
                "generate_child_state",
                "generate_child_states",
                "legal_actions",
                "is_final_state",
                "evaluate_final_state",
            ]
            for method in cached_methods:
                cache_info = getattr(game_manager, method).cache_info()
                hit_ratio = cache_info.hits / (cache_info.hits + cache_info.misses)
                print(f"{method}: {hit_ratio * 100:.1f}%")


def get_new_games(
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S, _A]]",
    self_playing_context: multiprocessing.SpawnContext,
    game_net: GameNet[_S, _A],
    block: bool = False,
) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    new_examples: List[Tuple[_S, Dict[_A, float], float]] = []
    game_lengths = []
    while True:
        block = block and not new_examples
        try:
            game = games_queue.get(block, timeout=1)
            game_lengths.append(len(game))
            new_examples.extend(game)
        except queue.Empty:
            if block and not self_playing_context.join(0):
                continue
            break
    if not new_examples:
        return []

    states, probability_targets, value_targets = zip(*new_examples)
    value_targets = torch.tensor(value_targets, dtype=torch.float32).reshape((-1, 1))
    assert value_targets.shape[0] == len(new_examples)
    probability_targets = game_net.distributions_to_tensor(states, probability_targets)
    assert probability_targets.shape[0] == len(new_examples)
    states = game_net.states_to_tensor(states)
    assert states.shape[0] == len(new_examples)
    new_examples = list(zip(states, probability_targets, value_targets))
    new_games = []
    i = 0
    for game_length in game_lengths:
        new_games.append(new_examples[i : i + game_length])
        i += game_length
    return new_games


def sample_replay_buffer(
    replay_buffer: List[TensorSelfPlayGame], batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    games = random.choices(
        replay_buffer, weights=[len(game) for game in replay_buffer], k=batch_size
    )
    examples = [random.choice(game) for game in games]
    states, probability_targets, value_targets = zip(*examples)
    states = torch.stack(states).to(device)
    probability_targets = torch.stack(probability_targets).to(device)
    value_targets = torch.stack(value_targets).to(device)
    return states, probability_targets, value_targets


def evaluate(
    game_net: GameNet[_S, _A],
    previous_game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    config: TrainingConfiguration,
) -> Tuple[AgentComparison, AgentComparison]:
    state_evaluator = cached_state_evaluator(game_net)
    previous_state_evaluator = cached_state_evaluator(previous_game_net)
    random_mcts_evaluation = compare_agents(
        (
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations,
                    config.rollout_policy,
                    state_evaluator=state_evaluator,
                )
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations * 4,
                    lambda s: random.choice(game_manager.legal_actions(s)),
                    state_evaluator=None,
                )
            ),
        ),
        config.evaluation_games,
        game_manager,
    )
    previous_evaluation = compare_agents(
        (
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations,
                    config.rollout_policy,
                    state_evaluator,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_factor=config.dirichlet_factor,
                ),
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations,
                    config.rollout_policy,
                    previous_state_evaluator,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_factor=config.dirichlet_factor,
                ),
            ),
        ),
        config.evaluation_games,
        game_manager,
    )
    return random_mcts_evaluation, previous_evaluation


def cached_state_evaluator(
    game_net: GameNet[_S, _A]
) -> Callable[[_S], Tuple[float, Dict[_A, float]]]:
    @lru_cache(2 ** 20)
    def inner(state: _S) -> Tuple[float, Dict[_A, float]]:
        return game_net.evaluate_state(state)

    return inner


# def spawn_evaluator(
#     game_manager: GameManager[_S, _A],
#     num_games: int,
#     num_simulations: int,
#     rollout_policy: Optional[Callable[[_S], _A]],
#     epsilon: float,
# ) -> Tuple[
#     multiprocessing.SpawnContext,
#     "multiprocessing.Queue[GameNet[_S, _A]]",
#     "multiprocessing.Queue[Tuple[float, AgentComparison, AgentComparison]]",
# ]:
#     game_net_queue: "multiprocessing.Queue[GameNet[_S, _A]]" = multiprocessing.Queue()
#     evaluation_results_queue: "multiprocessing.Queue[Tuple[float, AgentComparison, AgentComparison]]" = multiprocessing.Queue()
#     context = multiprocessing.spawn(
#         evaluator,
#         (
#             game_manager,
#             num_simulations,
#             rollout_policy,
#             game_net_queue,
#             evaluation_results_queue,
#             epsilon,
#         ),
#         nprocs=1,
#         join=False,
#     )
#     return context, game_net_queue, evaluation_results_queue
#
#
# def evaluator(
#     process_number: int,
#     game_manager: GameManager[_S, _A],
#     num_simulations: int,
#     rollout_policy: Optional[Callable[[_S], _A]],
#     game_net_queue: "multiprocessing.Queue[GameNet[_S, _A]]",
#     results_queue: "multiprocessing.Queue[Tuple[float, AgentComparison, AgentComparison]]",
#     epsilon: float,
# ) -> None:
#     previous_game_net = game_net_queue.get()
#     previous_game_net.to(torch.device("cuda:1"))
#     prev_evaluation_time = time.perf_counter()
#     while True:
#         try:
#             game_net = game_net_queue.get()
#         except OSError:
#             break
#         game_net.to(torch.device("cuda:1"))
#         print(f"{time.strftime('%H:%M:%S')} evaluating")
#         random_mcts_evaluation, previous_evaluation = evaluate(
#             game_net,
#             previous_game_net,
#             game_manager,
#             num_simulations,
#             rollout_policy,
#             epsilon,
#         )
#         print(f"{time.strftime('%H:%M:%S')} done evaluating")
#         previous_game_net = game_net
#         results_queue.put(
#             (
#                 time.perf_counter() - prev_evaluation_time,
#                 previous_evaluation,
#                 random_mcts_evaluation,
#             )
#         )
#         prev_evaluation_time = time.perf_counter()
