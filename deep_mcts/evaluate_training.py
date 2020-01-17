from pathlib import Path
import random
import json
from typing import Type, TypeVar, Optional
from multiprocessing import Pool

import pandas as pd
import torch

from deep_mcts.game import State, GameManager
from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.train import cached_state_evaluator
from deep_mcts.tournament import tournament, compare_agents
from deep_mcts.gamenet import GameNet


_S = TypeVar("_S", bound=State)


def evaluate_training(
    save_dir: Path,
    net_class: Type[GameNet[_S]],
    manager: GameManager[_S],
    device: torch.device,
) -> None:
    save_dirs = sorted(dir for dir in save_dir.iterdir())[-20:]
    with Pool(processes=10) as pool:
        pool.starmap(
            evaluate_models,
            [(save_dir, net_class, manager, device) for save_dir in save_dirs],
        )


def evaluate_models(
    model_dir: Path,
    net_class: Type[GameNet[_S]],
    manager: GameManager[_S],
    device: torch.device,
) -> None:
    model_files = sorted(
        [f for f in model_dir.iterdir() if f.name.endswith(".tar")],
        key=lambda f: int(f.name[5:-4]),
    )
    models = [
        net_class.from_path_full(str(model_file), manager) for model_file in model_files  # type: ignore[arg-type]
    ]
    random_agent = MCTSAgent(
        MCTS(
            manager,
            num_simulations=100,
            rollout_policy=lambda s: random.choice(manager.legal_actions(s)),
            state_evaluator=None,
        )
    )
    print(model_dir.name)
    previous_agent: Optional[MCTSAgent[_S]] = None
    results = []
    for i, model in enumerate(models):
        print(model_dir.name, model_files[i].name)
        model.to(device)
        state_evaluator = cached_state_evaluator(model)
        agent = MCTSAgent(
            MCTS(
                manager,
                num_simulations=50,
                rollout_policy=None,
                state_evaluator=state_evaluator,
            ),
            epsilon=0.05,
        )
        if previous_agent is None:
            previous_evaluation = (
                (0.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
            )
        else:
            previous_evaluation = compare_agents(
                (agent, previous_agent), num_games=40, game_manager=manager
            )
        previous_agent = agent
        random_evaluation = compare_agents(
            (
                MCTSAgent(
                    MCTS(
                        manager,
                        num_simulations=50,
                        rollout_policy=None,
                        state_evaluator=state_evaluator,
                    )
                ),
                random_agent,
            ),
            num_games=40,
            game_manager=manager,
        )
        training_iterations = int(model_files[i].name[5:-4])
        results.append((training_iterations, random_evaluation, previous_evaluation))
    evaluations = pd.DataFrame(
        [
            [
                training_iterations,
                *[*random_evaluation[0], *random_evaluation[1], *random_evaluation[2],],
                *[
                    *previous_evaluation[0],
                    *previous_evaluation[1],
                    *previous_evaluation[2],
                ],
            ]
            for (
                training_iterations,
                random_evaluation,
                previous_evaluation,
            ) in results
        ],
        columns=[
            "training_iterations",
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
    result_dir = model_dir.parent.parent / "training"
    result_dir.mkdir(exist_ok=True)
    evaluations.to_csv(str(result_dir / f"{model_dir.name}.csv"))
