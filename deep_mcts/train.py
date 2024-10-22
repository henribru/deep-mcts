import queue
import random
import textwrap
import time
from functools import lru_cache
import json
from typing import (
    Iterable,
    Tuple,
    Optional,
    TypeVar,
    List,
    cast,
    Sequence,
    Generic,
    Dict,
    Any,
)

import pandas as pd
import torch
import torch.autograd
from dataclasses import dataclass, asdict
from torch import multiprocessing

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
    batch_size: int = 1024
    replay_buffer_max_size: int = 100_000
    train_device: torch.device = torch.device("cuda:1")
    self_play_device: torch.device = torch.device("cuda:0")
    evaluation_games: int = 20
    transfer_interval: int = 1000

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        rollout_policy = d.pop("rollout_policy")
        d["has_rollout_policy"] = rollout_policy is not None
        for device in ["train_device", "self_play_device"]:
            device_type = d[device].type
            device_index = d[device].index
            d[
                device
            ] = f"{device_type}{f':{device_index}' if device_index is not None else ''}"
        return d


def train(game_net: GameNet[_S], config: TrainingConfiguration[_S]) -> None:
    evaluations = pd.DataFrame(
        [
            [
                training_iterations,
                training_games,
                *[*random_evaluation[0], *random_evaluation[1], *random_evaluation[2]],
                *[
                    *previous_evaluation[0],
                    *previous_evaluation[1],
                    *previous_evaluation[2],
                ],
            ]
            for (
                training_iterations,
                training_games,
                random_evaluation,
                previous_evaluation,
            ) in _train(game_net, config)
        ],
        columns=[
            "training_iterations",
            "training_games",
            "random_first_wins",
            "random_second_wins",
            "random_first_draws",
            "random_second_draws",
            "random_first_losses",
            "random_second_losses",
            "previous_first_wins",
            "previous_second_wins",
            "previous_first_draws",
            "previous_second_draws",
            "previous_first_losses",
            "previous_second_losses",
        ],
    )
    evaluations.to_csv(f"{config.save_dir}/evaluations.csv")


def _train(
    game_net: GameNet[_S], config: TrainingConfiguration[_S]
) -> Iterable[Tuple[int, int, AgentComparison, AgentComparison]]:
    print(f"{time.strftime('%H:%M:%S')} Starting")
    game_manager = game_net.manager
    game_net.to(config.train_device)
    replay_buffer: List[TensorSelfPlayGame] = []
    game_net.save_full(f"{config.save_dir}/anet-0.tar")
    with open(f"{config.save_dir}/parameters.json", "w") as f:
        net_parameters = game_net.parameters()
        net_parameters.pop("state_dict")
        net_parameters["optimizer_cls"] = net_parameters["optimizer_cls"].__name__
        parameters = {
            "net_parameters": net_parameters,
            "training_configuration": config.to_json_dict(),
        }
        json.dump(parameters, f, indent=2)
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn")
    self_play_game_net = game_net.copy().to(config.self_play_device)
    last_trained_iteration = torch.tensor([-1])
    self_playing_context, games_queue = spawn_self_play_example_creators(
        self_play_game_net, last_trained_iteration, config
    )
    previous_game_net = game_net.copy().to(config.train_device)
    training_iterations = 0
    training_games_count = 0
    training_examples_count = 0
    policy_loss = torch.tensor([0.0], device=config.train_device)
    value_loss = torch.tensor([0.0], device=config.train_device)
    accuracy = torch.tensor([0.0], device=config.train_device)
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
            filepath = f"{config.save_dir}/anet-{training_iterations + 1}.tar"
            game_net.save_full(filepath)
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
            yield training_iterations, training_games_count, random_mcts_evaluation, previous_evaluation

        training_iterations += 1


def spawn_self_play_example_creators(
    game_net: GameNet[_S],
    last_trained_iteration: torch.Tensor,
    config: TrainingConfiguration[_S],
) -> Tuple[multiprocessing.SpawnContext, "multiprocessing.Queue[SelfPlayGame[_S]]"]:
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S]]" = multiprocessing.Queue()
    context = multiprocessing.spawn(
        create_self_play_examples,
        (game_net, last_trained_iteration, config, games_queue),
        nprocs=config.nprocs,
        join=False,
    )
    return context, games_queue


def create_self_play_examples(
    process_number: int,
    game_net: GameNet[_S],
    last_trained_iteration: torch.Tensor,
    config: TrainingConfiguration[_S],
    games_queue: "multiprocessing.Queue[SelfPlayGame[_S]]",
) -> None:
    game_manager = game_net.manager
    last_cached_iteration = 0
    # Use a uniform evaluator as the starting point
    def uniform_state_evaluator(state: _S) -> Tuple[float, List[float]]:
        legal_actions = set(game_manager.legal_actions(state))
        return (
            0.5,
            [
                1 / len(legal_actions) if action in legal_actions else 0.0
                for action in range(game_manager.num_actions)
            ],
        )

    state_evaluator: StateEvaluator[_S] = uniform_state_evaluator
    for i in range(config.num_games):
        # Recreate the cache if the network has been trained since
        # we last created the cache
        last_trained_iteration_value = cast(int, last_trained_iteration.item())
        if last_trained_iteration_value > last_cached_iteration:
            state_evaluator = cached_state_evaluator(game_net)
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
    self_playing_context: multiprocessing.SpawnContext,
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
            if block and not self_playing_context.join(0):
                continue
            break
    if not new_examples:
        return []

    states, probability_targets, value_targets = zip(*new_examples)
    value_targets = torch.tensor(value_targets, dtype=torch.float32).reshape((-1, 1))
    assert value_targets.shape[0] == len(new_examples)
    probability_targets = game_net.distributions_to_tensor(probability_targets)
    assert probability_targets.shape[0] == len(new_examples)
    states = game_net.states_to_tensor(states)
    assert states.shape[0] == len(new_examples)
    new_examples = list(
        zip(states, probability_targets, value_targets)  # type: ignore[call-overload]
    )
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
                    config.num_simulations * 2,
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
                ),
                epsilon=config.epsilon,
            ),
            MCTSAgent(
                MCTS(
                    game_manager,
                    config.num_simulations,
                    config.rollout_policy,
                    previous_state_evaluator,
                ),
                epsilon=config.epsilon,
            ),
        ),
        config.evaluation_games,
        game_manager,
    )
    return random_mcts_evaluation, previous_evaluation


def cached_state_evaluator(game_net: GameNet[_S]) -> StateEvaluator[_S]:
    @lru_cache(2 ** 20)
    def inner(state: _S) -> Tuple[float, Sequence[float]]:
        value, probabilities = game_net.evaluate(state)
        return value, probabilities.tolist()

    return inner
