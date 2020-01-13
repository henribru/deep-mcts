from pathlib import Path
import re
import random
import json
from typing import Type, TypeVar

import torch

from deep_mcts.othello.convolutionalnet import ConvolutionalOthelloNet
from deep_mcts.othello.game import OthelloManager
from deep_mcts.game import State, GameManager
from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.train import cached_state_evaluator
from deep_mcts.tournament import tournament
from deep_mcts.gamenet import GameNet

_S = TypeVar("_S", bound=State)


def evaluate_simple_rollouts(
    save_dir: Path, net_class: Type[GameNet[_S]], manager: GameManager[_S]
) -> None:
    save_dirs = sorted(dir for dir in save_dir.iterdir())[-20:]
    model_files = [
        max(
            (f for f in save_dir.iterdir() if f.name.endswith(".tar")),
            key=lambda f: int(f.name[5:-4]),
        )
        for save_dir in save_dirs
    ]

    models = [
        net_class.from_path_full(str(model_file)).to(torch.device("cuda:0"))
        for model_file in model_files
    ]
    agents = [
        [
            MCTSAgent(
                MCTS(
                    manager,
                    num_simulations=50,
                    rollout_policy=(
                        lambda state: random.choice(manager.legal_actions(state))
                    )
                    if i > 0
                    else None,
                    state_evaluator=cached_state_evaluator(model),
                    rollout_share=i / 100,
                )
            )
            for i in range(0, 101, 10)
        ]
        for model in models
    ]
    result_dir = Path(__file__).resolve().parent / "simple_rollouts"
    result_dir.mkdir(exist_ok=True)
    for i, sub_agents in enumerate(agents):
        print(i)
        results = tournament(sub_agents, num_games=40, game_manager=manager)
        with open(result_dir / f"{save_dirs[i].name}.json", "w") as f:
            json.dump(results, f)
