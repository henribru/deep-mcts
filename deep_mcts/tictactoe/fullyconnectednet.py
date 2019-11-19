from typing import Tuple, Sequence, Mapping, TYPE_CHECKING, Type, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim.optimizer

from deep_mcts.game import Player, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.tictactoe.game import (
    TicTacToeState,
    Action,
    TicTacToeManager,
    CellState,
)

if TYPE_CHECKING:
    TensorPairModule = nn.Module[Tuple[torch.Tensor, torch.Tensor]]
else:
    TensorPairModule = nn.Module


class FullyConnectedTicTacToeModule(TensorPairModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 ** 2 * 2 + 1, 128)
        self.value_head = nn.Linear(128, 1)
        self.policy_head = nn.Linear(128, 3 ** 2)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = x
        assert input.shape == (input.shape[0], 3 ** 2 * 2 + 1)
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        assert policy.shape == (input.shape[0], (input.shape[1] - 1) / 2)
        value = self.value_head(x)
        assert value.shape == (input.shape[0], 1)
        return value, policy


class FullyConnectedTicTacToeNet(GameNet[TicTacToeState]):
    grid_size: int

    def __init__(
        self,
        manager: Optional[GameManager[TicTacToeState]] = None,
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"] = torch.optim.SGD,
        optimizer_args: Tuple[Any, ...] = (),
        optimizer_kwargs: Mapping[str, Any] = {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.001,
        },
    ) -> None:
        super().__init__(
            FullyConnectedTicTacToeModule(),
            manager if manager is not None else TicTacToeManager(),
            optimizer_cls,
            optimizer_args,
            optimizer_kwargs,
        )

    def _mask_illegal_moves(
        self, states: Sequence[TicTacToeState], output: torch.Tensor
    ) -> torch.Tensor:
        states = torch.tensor([state.grid for state in states]).reshape(-1, 9)
        legal_moves = (states == CellState.EMPTY).to(
            dtype=torch.float32, device=self.device
        )
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    def states_to_tensor(self, states: Sequence[TicTacToeState]) -> torch.Tensor:
        players = torch.tensor([state.player for state in states]).reshape(
            (len(states), -1)
        )
        grids = torch.tensor([state.grid for state in states]).reshape(
            (len(states), -1)
        )
        for i in range(len(states)):
            assert (grids[i, :] == torch.tensor(states[i].grid).flatten()).all()
        first_player = grids == Player.FIRST
        second_player = grids == Player.SECOND
        assert ((first_player.sum(dim=1) - second_player.sum(dim=1)) <= 1).all()
        # TODO: Make current player come first instead of always player 0?
        tensor = torch.cat((first_player, second_player, players), dim=1).to(
            dtype=torch.float32
        )
        assert tensor.shape == (len(states), 2 * 3 ** 2 + 1)
        return tensor
        # players = np.array([state.player for state in states]).reshape(
        #     (len(states), -1)
        # )
        # grids = np.stack([state.grid for state in states]).reshape((len(states), -1))
        # for i in range(len(states)):
        #     assert np.all(grids[i, :] == np.array(states[i].grid).flatten())
        # current_player = grids == np.array([state.player for state in states]).reshape(
        #     (-1, 1)
        # )
        # other_player = grids == np.array(
        #     [0 if state.player == 1 else 1 for state in states]
        # ).reshape((-1, 1))
        # assert np.all((current_player.sum(axis=1) - other_player.sum(axis=1)) <= 1)
        # grids = np.concatenate((current_player, other_player, players), axis=1)
        # tensor = torch.as_tensor(grids, dtype=torch.float32, device=self.device)
        # assert tensor.shape == (len(grids), 2 * 3 ** 2 + 1)
        # return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[TicTacToeState],
        distributions: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        targets = torch.tensor(distributions, dtype=torch.float32)
        assert targets.shape == (len(distributions), 3 ** 2)
        return targets

    def copy(self) -> "FullyConnectedTicTacToeNet":
        net = FullyConnectedTicTacToeNet(
            self.manager, self.optimizer_cls, self.optimizer_args, self.optimizer_kwargs
        )
        net.net.load_state_dict(self.net.state_dict())
        return net
