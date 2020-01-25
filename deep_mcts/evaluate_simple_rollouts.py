from pathlib import Path
import re
import random
import json
from typing import Type, TypeVar
from multiprocessing import Pool
import torch

from deep_mcts.game import State, GameManager
from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.train import cached_state_evaluator
from deep_mcts.tournament import tournament
from deep_mcts.gamenet import GameNet

_S = TypeVar("_S", bound=State)


def evaluate_simple_rollouts(
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
    model_file = max(
        (f for f in model_dir.iterdir() if f.name.endswith(".tar")),
        key=lambda f: int(f.name[5:-4]),
    )

    model = net_class.from_path_full(str(model_file), manager).to(device)
    state_evaluator = cached_state_evaluator(model)
    agents = [
        MCTSAgent(
            MCTS(
                manager,
                num_simulations=50,
                rollout_policy=(
                    lambda state: random.choice(manager.legal_actions(state))
                )
                if i > 0
                else None,
                state_evaluator=state_evaluator,
                rollout_share=i / 100,
            )
        )
        for i in range(0, 101, 20)
    ]

    result_dir = model_dir.parent.parent / "simple_rollouts"
    result_dir.mkdir(exist_ok=True)
    print(model_dir.name)
    results = tournament(agents, num_games=40, game_manager=manager)
    with open(result_dir / f"{model_dir.name}.json", "w") as f:
        json.dump(results, f)
