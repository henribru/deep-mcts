import queue
import random
import textwrap
import time
from functools import lru_cache
from typing import Iterable, Tuple, Optional, TypeVar, List, cast, Sequence, Generic

import pandas as pd
import torch
import torch.autograd
from dataclasses import dataclass
import torch.multiprocessing
import multiprocessing

from deep_mcts.game import State, GameManager, Player
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS, MCTSAgent, StateEvaluator, RolloutPolicy
from deep_mcts.tournament import compare_agents, AgentComparison

_S = TypeVar("_S", bound=State)
SelfPlayExample = Tuple[_S, Sequence[float], float]
SelfPlayGame = List[SelfPlayExample[_S]]
TensorSelfPlayExample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TensorSelfPlayGame = List[TensorSelfPlayExample]


@dataclass(frozen=True)
class TrainingConfiguration(Generic[_S]):
    num_games: int
    num_simulations: int
    save_interval: int
    evaluation_interval: int
    save_dir: str
    sample_move_cutoff: int
    dirichlet_alpha: float
    dirichlet_factor: float = 0.25
    rollout_policy: Optional[RolloutPolicy[_S]] = None
    epsilon: float = 0.05
    nprocs: int = 25
    batch_size: int = 512
    self_play_batch_size: int = 8
    replay_buffer_max_size: int = 100_000
    train_device: torch.device = torch.device("cuda:1")
    self_play_device: torch.device = torch.device("cuda:0")
    evaluation_games: int = 20
    transfer_interval: int = 1000


def train(game_net: GameNet[_S], config: TrainingConfiguration[_S]) -> None:
    evaluations = pd.DataFrame.from_dict(
        {
            i: (random_evaluation, previous_evaluation)
            for i, random_evaluation, previous_evaluation in _train(game_net, config)
        },
        orient="index",
        columns=["against_random", "against_previous"],
    )
    evaluations.to_csv(f"{config.save_dir}/evaluations.csv")


def _train(
    game_net: GameNet[_S], config: TrainingConfiguration[_S]
) -> Iterable[Tuple[int, AgentComparison, AgentComparison]]:
    print(f"{time.strftime('%H:%M:%S')} Starting")
    game_manager = game_net.manager
    game_net.to(config.train_device)
    replay_buffer: List[TensorSelfPlayGame] = []
    game_net.save(f"{config.save_dir}/anet-0.pth")
    multiprocessing.set_start_method("spawn")
    self_play_game_net = game_net.copy().to(config.self_play_device)
    last_trained_iteration = torch.tensor([-1])
    (
        evaluation_queue,
        result_receive_pipes,
        self_play_evaluator_context,
    ) = spawn_self_play_evaluator(self_play_game_net, config)
    self_playing_context, games_queue = spawn_self_play_example_creators(
        game_manager,
        last_trained_iteration,
        config,
        evaluation_queue,
        result_receive_pipes,
    )
    previous_game_net = game_net.copy().to(config.train_device)
    training_iterations = 0
    training_games_count = 0
    training_examples_count = 0
    policy_loss = torch.tensor([0.0], device=config.train_device)
    value_loss = torch.tensor([0.0], device=config.train_device)
    accuracy = torch.tensor([0.0], device=config.train_device)
    prev_evaluation_time = time.perf_counter()
    while not self_playing_context.join(0) and not self_play_evaluator_context.join(0):
        new_games = get_new_games(
            games_queue, self_playing_context, self_play_evaluator_context, game_net, block=len(replay_buffer) == 0
        )
        training_games_count += len(new_games)
        training_examples_count += sum(len(game) for game in new_games)
        replay_buffer.extend(new_games)
        if len(replay_buffer) > config.replay_buffer_max_size:
            replay_buffer = replay_buffer[-config.replay_buffer_max_size :]
        states, probability_targets, value_targets = sample_replay_buffer(
            replay_buffer, config.batch_size, config.train_device
        )

        if __debug__:
            with torch.autograd.detect_anomaly():
                batch_policy_loss, batch_value_loss, batch_accuracy = game_net.train(
                    states, probability_targets, value_targets
                )
        else:
            batch_policy_loss, batch_value_loss, batch_accuracy = game_net.train(
                states, probability_targets, value_targets
            )
        policy_loss += batch_policy_loss
        value_loss += batch_value_loss
        accuracy += batch_accuracy

        if (training_iterations + 1) % config.transfer_interval == 0:
            self_play_game_net.load_state_dict(game_net.net.state_dict())
            last_trained_iteration[0] = training_iterations

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
                game_net, previous_game_net, game_manager, config
            )
            print(
                textwrap.dedent(
                    f"""
                    {time.strftime('%H:%M:%S')}
                    iterations: {training_iterations + 1} 
                    games: {training_games_count}
                    examples: {training_examples_count} 
                    evaluation_duration: {time.perf_counter() - prev_evaluation_time:.2f}
                    previous: {previous_evaluation} 
                    random MCTS: {random_mcts_evaluation}
                    policy_loss: {policy_loss.item() / config.evaluation_interval:.2f}
                    value_loss: {value_loss.item() / config.evaluation_interval:.2f}
                    accuracy: {accuracy.item() / config.evaluation_interval:.2f}
                    """
                )
            )
            policy_loss[0] = 0.0
            value_loss[0] = 0.0
            accuracy[0] = 0.0
            prev_evaluation_time = time.perf_counter()
            previous_game_net.load_state_dict(game_net.net.state_dict())
            yield training_iterations, random_mcts_evaluation, previous_evaluation

        training_iterations += 1


def spawn_self_play_evaluator(
    game_net: GameNet[_S], config: TrainingConfiguration[_S]
) -> Tuple[
    "multiprocessing.Queue[Tuple[int, _S]]",
    Sequence["multiprocessing.connection.Connection"],
    torch.multiprocessing.SpawnContext,
]:
    evaluation_queue: "multiprocessing.Queue[Tuple[int, _S]]" = multiprocessing.Queue()
    result_pipes: List[
        Tuple[
            multiprocessing.connection.Connection, multiprocessing.connection.Connection
        ]
    ] = [multiprocessing.Pipe() for _ in range(config.nprocs)]
    result_receive_pipes, result_send_pipes = zip(*result_pipes)
    context: torch.multiprocessing.SpawnContext = torch.multiprocessing.spawn(
        self_play_evaluator,
        (evaluation_queue, result_send_pipes, game_net, config.self_play_batch_size),
        join=False
    )
    return evaluation_queue, result_receive_pipes, context


def self_play_evaluator(
    process_number: int,
    evaluation_queue: "multiprocessing.Queue[Tuple[int, _S]]",
    result_pipes: Sequence["multiprocessing.connection.Connection"],
    game_net: GameNet[_S],
    batch_size: int,
) -> None:
    while True:
        batch = [evaluation_queue.get() for _ in range(batch_size)]
        process_numbers, batch = zip(*batch)
        values, probabilities = game_net.forward(batch)
        for i, process_number in enumerate(process_numbers):
            result_pipes[process_number].send((i, values, probabilities))


def spawn_self_play_example_creators(
    game_manager: GameManager[_S],
    last_trained_iteration: torch.Tensor,
    config: TrainingConfiguration[_S],
    evaluation_queue: "multiprocessing.Queue[Tuple[int, _S]]",
    result_receive_pipes: Sequence["multiprocessing.connection.Connection"],
) -> Tuple[
    torch.multiprocessing.SpawnContext, "multiprocessing.Queue[SelfPlayGame[_S]]"
]:
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S]]" = multiprocessing.Queue()
    context = torch.multiprocessing.spawn(
        create_self_play_examples,
        (
            game_manager,
            last_trained_iteration,
            config,
            games_queue,
            evaluation_queue,
            result_receive_pipes,
        ),
        nprocs=config.nprocs,
        join=False,
    )
    return context, games_queue


def create_self_play_examples(
    process_number: int,
    game_manager: GameManager[_S],
    last_trained_iteration: torch.Tensor,
    config: TrainingConfiguration[_S],
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S]]",
    evaluation_queue: "multiprocessing.Queue[Tuple[int, _S]]",
    result_receive_pipes: Sequence["multiprocessing.connection.Connection"],
) -> None:
    last_cached_iteration = 0
    # Use a uniform evaluator as the starting point
    def uniform_state_evaluator(state: _S) -> Tuple[float, Sequence[float]]:
        legal_actions = set(game_manager.legal_actions(state))
        return (
            0.5,
            [
                1 / len(legal_actions) if action in legal_actions else 0.0
                for action in range(game_manager.num_actions)
            ],
        )

    def batched_state_evaluator(state: _S) -> Tuple[float, Sequence[float]]:
        evaluation_queue.put((process_number, state))
        i, values, probabilities = result_receive_pipes[process_number].recv()
        return values[i].item(), probabilities[i].tolist()

    state_evaluator: StateEvaluator[_S] = uniform_state_evaluator
    for i in range(config.num_games):
        # Recreate the cache if the network has been trained since
        # we last created the cache
        last_trained_iteration_value = cast(int, last_trained_iteration.item())
        if last_trained_iteration_value > last_cached_iteration:
            state_evaluator = cached_state_evaluator(batched_state_evaluator)
            last_cached_iteration = last_trained_iteration_value
        mcts = MCTS(
            game_manager,
            config.num_simulations,
            config.rollout_policy,
            state_evaluator,
            config.sample_move_cutoff,
            config.dirichlet_alpha,
            config.dirichlet_factor,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.self_play():
            examples.append((state, visit_distribution))
        # The network uses a range of [-1, 1]
        outcome = (
            cast(float, game_manager.evaluate_final_state(next_state).value) * 2 - 1
        )
        games_queue.put(
            [
                (
                    state,
                    visit_distribution,
                    outcome if state.player == Player.max_player() else -outcome,
                )
                for state, visit_distribution in examples
            ]
        )
        if i % 100 == 0 and process_number == 0:
            print(f"{time.strftime('%H:%M:%S')} {i}")
            # cached_methods = [
            #     "generate_child_state",
            #     "generate_child_states",
            #     "legal_actions",
            #     "is_final_state",
            #     "evaluate_final_state",
            # ]
            # for method in cached_methods:
            #     cache_info = getattr(game_manager, method).cache_info()
            #     hit_ratio = cache_info.hits / (cache_info.hits + cache_info.misses)
            #     print(f"{method}: {hit_ratio * 100:.1f}%")


def get_new_games(
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S]]",
    self_playing_context: torch.multiprocessing.SpawnContext,
    self_play_evaluator_context: torch.multiprocessing.SpawnContext,
    game_net: GameNet[_S],
    block: bool = False,
) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    new_examples: List[SelfPlayExample[_S]] = []
    game_lengths = []
    while True:
        block = block and not new_examples
        try:
            game = games_queue.get(block, timeout=1)
            game_lengths.append(len(game))
            new_examples.extend(game)
        except queue.Empty:
            if block and not self_playing_context.join(0) and not self_play_evaluator_context.join(0):
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
    game_net: GameNet[_S],
    previous_game_net: GameNet[_S],
    game_manager: GameManager[_S],
    config: TrainingConfiguration[_S],
) -> Tuple[AgentComparison, AgentComparison]:
    state_evaluator = cached_state_evaluator(game_net.forward_single)
    previous_state_evaluator = cached_state_evaluator(game_net.forward_single)
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
                )
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations,
                    config.rollout_policy,
                    previous_state_evaluator,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_factor=config.dirichlet_factor,
                )
            ),
        ),
        config.evaluation_games,
        game_manager,
    )
    return random_mcts_evaluation, previous_evaluation


def cached_state_evaluator(state_evaluator: StateEvaluator[_S]) -> StateEvaluator[_S]:
    @lru_cache(2 ** 20)
    def inner(state: _S) -> Tuple[float, Sequence[float]]:
        return state_evaluator(state)

    return inner
