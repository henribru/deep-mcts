import queue
import random
import time
from typing import Iterable, Tuple, Optional, TypeVar, Callable, Deque, Dict, List

import torch
from torch import multiprocessing

from deep_mcts.game import State, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS, MCTSAgent, Player
from deep_mcts.tournament import compare_agents, AgentComparison

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A")


def train(
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    save_interval: int,
    evaluation_interval: int,
    save_dir: str,
    rollout_policy: Optional[Callable[[_S], _A]] = None,
    epsilon: float = 0.05,
    nprocs: int = 25,
    batch_size: int = 512,
    replay_buffer_max_size: int = 100_000,
) -> Iterable[Tuple[int, AgentComparison, AgentComparison]]:
    print(f"{time.strftime('%H:%M:%S')} Starting")
    device = torch.device("cuda:1")
    game_net.to(device)
    replay_buffer = Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]](
        [], replay_buffer_max_size
    )
    game_net.save(f"{save_dir}/anet-0.pth")
    multiprocessing.set_start_method("spawn")
    self_playing_context, games_queue = spawn_self_play_example_creators(
        game_net,
        game_manager,
        num_games,
        num_simulations,
        rollout_policy,
        epsilon,
        nprocs,
        device=torch.device("cuda:0"),
    )
    previous_game_net = game_net.copy()
    training_iterations = 0
    training_games_count = 0
    training_examples_count = 0
    prev_evaluation_time = time.perf_counter()
    while not self_playing_context.join(0):
        new_games_count, new_examples = get_new_examples(
            games_queue, self_playing_context, game_net, block=len(replay_buffer) == 0
        )
        training_games_count += new_games_count
        training_examples_count += len(new_examples)
        replay_buffer.extend(new_examples)
        states, probability_targets, value_targets = sample_replay_buffer(
            replay_buffer, batch_size, device
        )
        game_net.train(states, probability_targets, value_targets)

        if save_interval != 0 and (training_iterations + 1) % save_interval == 0:
            filepath = f"{save_dir}/anet-{training_iterations + 1}.pth"
            game_net.save(filepath)
            print(f"{time.strftime('%H:%M:%S')} Saved {filepath}")

        if (
            evaluation_interval != 0
            and (training_iterations + 1) % evaluation_interval == 0
        ):
            print(f"{time.strftime('%H:%M:%S')} evaluating")
            random_mcts_evaluation, previous_evaluation = evaluate(
                game_net,
                previous_game_net,
                game_manager,
                num_simulations,
                rollout_policy,
                epsilon,
            )
            print(
                f"{time.strftime('%H:%M:%S')} "
                f"iterations: {training_iterations} games: {training_games_count} "
                f"examples: {training_examples_count} evaluation_duration: {time.perf_counter() - prev_evaluation_time:.2f} "
                f"previous: {previous_evaluation} random MCTS: {random_mcts_evaluation} "
            )
            prev_evaluation_time = time.perf_counter()
            previous_game_net = game_net.copy()
            yield training_iterations, random_mcts_evaluation, previous_evaluation

        training_iterations += 1


def spawn_self_play_example_creators(
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    rollout_policy: Optional[Callable[[_S], _A]],
    epsilon: float,
    nprocs: int,
    device: Optional[torch.device] = None,
) -> Tuple[
    multiprocessing.SpawnContext,
    "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]",
]:
    games_queue: "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]" = multiprocessing.Queue()
    context = multiprocessing.spawn(
        create_self_play_examples,
        (
            game_net,
            game_manager,
            num_games,
            num_simulations,
            rollout_policy,
            games_queue,
            epsilon,
            device,
        ),
        nprocs=nprocs,
        join=False,
    )
    return context, games_queue


def create_self_play_examples(
    process_number: int,
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    rollout_policy: Optional[Callable[[_S], _A]],
    games_queue: "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]",
    epsilon: float,
    device: Optional[torch.device],
) -> None:
    if device is not None:
        original_game_net = game_net
        game_net = game_net.copy()
        game_net.to(device)
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
                (
                    state,
                    visit_distribution,
                    outcome if state.player == Player.SECOND else -outcome,
                )
                for state, visit_distribution in examples
            ]
        )
        if device is not None:
            game_net.net.load_state_dict(original_game_net.net.state_dict())
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


def get_new_examples(
    games_queue: "multiprocessing.Queue[List[Tuple[_S, Dict[_A, float], float]]]",
    self_playing_context: multiprocessing.SpawnContext,
    game_net: GameNet[_S, _A],
    block: bool = False,
) -> Tuple[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    new_examples: List[Tuple[_S, Dict[_A, float], float]] = []
    games_count = 0
    while True:
        block = block and not new_examples
        try:
            examples = games_queue.get(block, timeout=1)
            games_count += 1
            new_examples.extend(examples)
        except queue.Empty:
            if block and not self_playing_context.join(0):
                continue
            break
    if not new_examples:
        return 0, []
    states, probability_targets, value_targets = zip(*new_examples)
    value_targets = torch.tensor(value_targets, dtype=torch.float32).reshape((-1, 1))
    assert value_targets.shape[0] == len(new_examples)
    probability_targets = game_net.distributions_to_tensor(states, probability_targets)
    assert probability_targets.shape[0] == len(new_examples)
    states = game_net.states_to_tensor(states)
    assert states.shape[0] == len(new_examples)
    return games_count, list(zip(states, probability_targets, value_targets))


def sample_replay_buffer(
    replay_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
    states, probability_targets, value_targets = zip(*examples)
    states = torch.stack(states).to(device)
    probability_targets = torch.stack(probability_targets).to(device)
    value_targets = torch.stack(value_targets).to(device)
    return states, probability_targets, value_targets


def evaluate(
    game_net: GameNet[_S, _A],
    previous_game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_simulations: int,
    rollout_policy: Optional[Callable[[_S], _A]],
    epsilon: float,
    games: int = 20,
) -> Tuple[AgentComparison, AgentComparison]:
    random_mcts_evaluation = compare_agents(
        (
            MCTSAgent(
                MCTS(
                    game_manager,
                    num_simulations,
                    rollout_policy,
                    state_evaluator=game_net.evaluate_state,
                )
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    num_simulations * 4,
                    lambda s: random.choice(game_manager.legal_actions(s)),
                    state_evaluator=None,
                )
            ),
        ),
        games,
        game_manager,
    )
    previous_evaluation = compare_agents(
        (
            MCTSAgent(
                MCTS(
                    game_manager,
                    num_simulations,
                    rollout_policy,
                    state_evaluator=game_net.evaluate_state,
                    epsilon=epsilon,
                )
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    num_simulations,
                    rollout_policy,
                    state_evaluator=previous_game_net.evaluate_state,
                    epsilon=epsilon,
                )
            ),
        ),
        games,
        game_manager,
    )
    return random_mcts_evaluation, previous_evaluation


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
