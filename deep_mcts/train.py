import random
import time
from typing import Iterable, Tuple, Optional, TypeVar, Callable, Deque, Dict

from deep_mcts.game import State, Action, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.mcts import MCTS, GreedyMCTSAgent
from deep_mcts.tournament import tournament, RandomAgent
from torch import multiprocessing
import queue

_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)


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
            examples_queue.put((state, visit_distribution, outcome if state.player == 0 else -outcome))


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
    game_net.save(f"anet-0.pth")
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
    now = time.time()
    example_queue: "multiprocessing.Queue[Tuple[_S, Dict[_A, float], float]]" = multiprocessing.Queue()
    game_net.net.share_memory()
    spawn_context = multiprocessing.spawn(
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
    while not spawn_context.join(0):
        while True:
            try:
                block = not replay_buffer
                replay_buffer.append(example_queue.get(block, timeout=1))
            except queue.Empty:
                if block and not spawn_context.join(0):
                    continue
                break
        examples = random.sample(replay_buffer, min(512, len(replay_buffer)))
        game_net.train(examples)
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
                time.time() - now,
                random_evaluation,
                previous_evaluation,
                original_evaluation,
                random_mcts_evaluation,
            )
            now = time.time()
            yield i + 1, random_evaluation, previous_evaluation
        if save_interval != 0 and (i + 1) % save_interval == 0:
            filepath = f"saves/anet-{i + 1}.pth"
            game_net.save(filepath)
            print(f"Saved {filepath}")
        i += 1
