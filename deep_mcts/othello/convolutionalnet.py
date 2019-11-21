from typing import Tuple, Mapping, Sequence, Type, Any, Optional

import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import GameManager, Action
from deep_mcts.gamenet import GameNet
from deep_mcts.othello.game import (
    OthelloState,
    OthelloManager,
)


class ConvolutionalOthelloNet(GameNet[OthelloState]):
    grid_size: int
    num_residual: int
    channels: int

    def __init__(
        self,
        grid_size: int,
        manager: Optional[GameManager[OthelloState]] = None,
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"] = torch.optim.SGD,
        optimizer_args: Tuple[Any, ...] = (),
        optimizer_kwargs: Mapping[str, Any] = {
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.001,
        },
        num_residual: int = 1,
        channels: int = 128,
    ) -> None:
        super().__init__(
            ConvolutionalNet(
                num_residual=num_residual,
                grid_size=grid_size,
                in_channels=3,
                channels=channels,
                policy_features=grid_size ** 2 + 1,
                policy_shape=(grid_size ** 2 + 1,),
            ),
            manager if manager is not None else OthelloManager(grid_size),
            optimizer_cls,
            optimizer_args,
            optimizer_kwargs,
        )
        self.grid_size = grid_size
        self.num_residual = num_residual
        self.channels = channels

    def states_to_tensor(self, states: Sequence[OthelloState]) -> torch.Tensor:
        players, opposite_players, grids = zip(
            *[(state.player, state.player.opposite(), state.grid) for state in states]
        )
        players = torch.tensor(players).reshape(-1, 1, 1)
        opposite_players = torch.tensor(opposite_players).reshape(-1, 1, 1)

        player_grids = torch.full(
            (len(states), self.grid_size, self.grid_size), fill_value=-1.0
        )
        player_grids[:] = players

        # We want everything to be from the perspective of the current player.
        grids = torch.tensor(grids)
        current_player = (grids == players).float()
        other_player = (grids == opposite_players).float()
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.stack([current_player, other_player, player_grids], dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def copy(self) -> "ConvolutionalOthelloNet":
        net = ConvolutionalOthelloNet(
            self.grid_size,
            self.manager,
            self.optimizer_cls,
            self.optimizer_args,
            self.optimizer_kwargs,
            self.num_residual,
            self.channels,
        )
        net.net.load_state_dict(self.net.state_dict())
        return net
