from pathlib import Path
import re
import random
import json
from typing import Type, TypeVar, List, Optional, Callable, Tuple, Sequence
from multiprocessing import Pool

import torch
import numpy as np

from deep_mcts.game import State, GameManager
from deep_mcts.mcts import MCTS, MCTSAgent
from deep_mcts.train import cached_state_evaluator
from deep_mcts.tournament import tournament, compare_agents
from deep_mcts.gamenet import GameNet

_S = TypeVar("_S", bound=State)


def evaluate_complex_rollouts(
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
    print(model_dir.name)
    model_file = max(
        (f for f in model_dir.iterdir() if f.name.endswith(".tar")),
        key=lambda f: int(f.name[5:-4]),
    )
    time_per_move = 1.0
    for with_state_evaluator, dir_name in [
        (True, "with_state_evaluator"),
        (False, "without_state_evaluator"),
    ]:
        manager_copy = manager.copy()
        model = net_class.from_path_full(str(model_file), manager_copy).to(device)  # type: ignore[arg-type]
        cached_model = cached_state_evaluator(model)
        complex_agent = MCTSAgent(
            MCTS(
                manager_copy,
                num_simulations=float("inf"),  # type: ignore[arg-type]
                rollout_policy=lambda state: np.argmax(  # type: ignore[no-any-return]
                    cached_model(state)[1]
                ),
                state_evaluator=cached_model if with_state_evaluator else None,
                rollout_share=1.0,
                time_per_move=time_per_move,
            ),
            reset_fn=reset_caches,
        )
        manager_copy = manager.copy()
        state_evaluator: Optional[Callable[[_S], Tuple[float, Sequence[float]]]]
        if with_state_evaluator:
            model = net_class.from_path_full(str(model_file), manager_copy).to(device)  # type: ignore[arg-type]
            state_evaluator = cached_state_evaluator(model)
        else:
            state_evaluator = None

        simple_agent = MCTSAgent(
            MCTS(
                manager_copy,
                num_simulations=float("inf"),  # type: ignore[arg-type]
                rollout_policy=lambda state: random.choice(
                    manager_copy.legal_actions(state)
                ),
                state_evaluator=state_evaluator,
                rollout_share=1.0,
                time_per_move=time_per_move,
            ),
            reset_fn=reset_caches,
        )
        agents = (complex_agent, simple_agent)
        result_dir = model_dir.parent.parent / "complex_rollouts" / dir_name
        result_dir.mkdir(exist_ok=True)
        results = compare_agents(agents, num_games=40, game_manager=manager)

        with open(result_dir / f"{model_dir.name}.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "average_complex_simulations": complex_agent.simulation_stats,
                    "average_simple_simulations": simple_agent.simulation_stats,
                },
                f,
            )


def reset_caches(agent: MCTSAgent[_S]) -> None:
    if hasattr(agent.mcts.state_evaluator, "cache_clear"):
        agent.mcts.state_evaluator.cache_clear()  # type: ignore[union-attr]
    for attr_name in dir(agent.mcts.game_manager):
        attr = getattr(agent.mcts.game_manager, attr_name)
        if callable(attr) and hasattr(attr, "cache_clear"):
            attr.cache_clear()
