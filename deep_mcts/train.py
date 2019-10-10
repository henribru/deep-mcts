import multiprocessing
import queue
import random
import time
from collections import deque
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple, TypeVar

import torch.multiprocessing
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from deep_mcts.game import Action, GameManager, State
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS, GreedyMCTSAgent
from deep_mcts.tournament import RandomAgent, tournament

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


class Dataset(IterableDataset):
    def __init__(self, example_queue, spawn_context, anet, batch_size):
        self.example_queue = example_queue
        self.replay_buffer = deque()
        self.spawn_context = spawn_context
        self.anet = anet
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            while True:
                try:
                    state, action_distribution, outcome = self.example_queue.get(False)
                except queue.Empty:
                    break
                value_target = torch.tensor([outcome], dtype=torch.float32).reshape(
                    (-1, 1)
                )
                probability_target = self.anet._distributions_to_tensor(
                    [state], [action_distribution]
                )[0]
                state = self.anet._state_to_tensor(state)[0]
                self.replay_buffer.append((state, probability_target, value_target))
            batch_size = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            states, probability_targets, value_targets = zip(*batch)
            states = torch.stack(states)
            probability_targets = torch.stack(probability_targets)
            value_targets = torch.cat(value_targets)
            assert value_targets.shape == (batch_size, 1)
            assert probability_targets.shape[0] == batch_size
            assert states.shape[0] == batch_size
            yield states, probability_targets, value_targets


def create_self_play_examples(
    process_number: int,
    game_net: GameNet[_S, _A],
    game_manager: GameManager[_S, _A],
    num_games: int,
    num_simulations: int,
    rollout_policy: Optional[Callable[[_S], _A]],
    examples_queue: "multiprocessing.Queue[Tuple[_S, Dict[_A, float], float]]",
) -> None:
    for i in range(num_games):
        mcts = MCTS(
            game_manager,
            num_simulations,
            rollout_policy,
            state_evaluator=game_net.evaluate_state,
        )
        examples = []
        for state, next_state, action, visit_distribution in mcts.self_play():
            examples.append((state, visit_distribution))
        outcome = game_manager.evaluate_final_state(next_state)
        for state, visit_distribution in examples:
            examples_queue.put(
                (state, visit_distribution, outcome if state.player == 0 else -outcome)
            )


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
    replay_buffer = Deque[Tuple[_S, Dict[_A, float], float]]([], 100_000)
    game_net.save(f"saves/anet-0.pth")
    random_opponent = RandomAgent(game_manager)
    original_opponent = game_net.copy()
    epsilon = 0.1
    original_agent = GreedyMCTSAgent(
        MCTS(
            game_manager,
            100,
            rollout_policy,
            state_evaluator=original_opponent.evaluate_state,
        ),
        epsilon,
    )
    previous_agent: Optional[GreedyMCTSAgent[_S, _A]] = None
    multiprocessing.set_start_method("spawn")
    example_queue: "multiprocessing.Queue[Tuple[_S, Dict[_A, float], float]]" = multiprocessing.Queue()
    # game_net.net.share_memory()
    spawn_context = torch.multiprocessing.spawn(
        create_self_play_examples,
        (
            game_net,
            game_manager,
            num_games,
            num_simulations,
            rollout_policy,
            example_queue,
        ),
        nprocs=1,
        join=False,
    )
    i = 0
    while example_queue.empty():
        spawn_context.join(0)
    prev_evaluation_time = time.time()
    prev_train_time = time.time()
    for states, probability_targets, value_targets in DataLoader(
        Dataset(example_queue, spawn_context, game_net, 512),
        batch_size=None,
        num_workers=1,
    ):
        game_net.train(states, probability_targets, value_targets)
        print(i + 1, time.time() - prev_train_time)
        prev_train_time = time.time()
        if evaluation_interval != 0 and (i + 1) % evaluation_interval == 0:
            mcts = MCTS(
                game_manager,
                100,
                rollout_policy,
                state_evaluator=game_net.evaluate_state,
            )
            agent = GreedyMCTSAgent(mcts, epsilon)
            random_evaluation = tournament(
                [random_opponent, GreedyMCTSAgent(mcts)], 20, game_manager
            )[1][0]
            random_mcts_evaluation = tournament(
                [
                    GreedyMCTSAgent(
                        MCTS(
                            game_manager,
                            100,
                            lambda s: random.choice(game_manager.legal_actions(s)),
                            lambda s: (
                                0,
                                {
                                    a: 1 / len(game_manager.legal_actions(s))
                                    for a in game_manager.legal_actions(s)
                                },
                            ),
                        )
                    ),
                    GreedyMCTSAgent(mcts),
                ],
                20,
                game_manager,
            )[1][0]
            original_evaluation = tournament([original_agent, agent], 20, game_manager)[
                1
            ][0]
            previous_evaluation = (
                tournament([previous_agent, agent], 20, game_manager)[1][0]
                if previous_agent is not None
                else None
            )
            previous_agent = GreedyMCTSAgent(
                MCTS(
                    game_manager,
                    100,
                    rollout_policy,
                    state_evaluator=game_net.copy().evaluate_state,
                ),
                epsilon,
            )
            print(
                i + 1,
                time.time() - prev_evaluation_time,
                random_evaluation,
                previous_evaluation,
                original_evaluation,
                random_mcts_evaluation,
            )
            prev_evaluation_time = time.time()
            yield i + 1, random_evaluation, previous_evaluation
        if save_interval != 0 and (i + 1) % save_interval == 0:
            filepath = f"saves/anet-{i + 1}.pth"
            game_net.save(filepath)
            print(f"Saved {filepath}")
        i += 1
        if spawn_context.join(0):
            break
