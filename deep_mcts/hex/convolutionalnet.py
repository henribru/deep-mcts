from typing import Mapping, Sequence, Tuple, Type, Any, Optional

import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import CellState, Player, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.hex.game import Action, HexManager, HexState


class ConvolutionalHexNet(GameNet[HexState]):
    grid_size: int
    num_residual: int
    channels: int

    def __init__(
        self,
        grid_size: int,
        manager: Optional[GameManager[HexState]] = None,
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
                policy_features=grid_size ** 2,
                policy_shape=(grid_size, grid_size),
            ),
            manager if manager is not None else HexManager(grid_size),
            optimizer_cls,
            optimizer_args,
            optimizer_kwargs,
        )
        self.grid_size = grid_size
        self.num_residual = num_residual
        self.channels = channels

    def _mask_illegal_moves(
        self, states: Sequence[HexState], output: torch.Tensor
    ) -> torch.Tensor:
        # Since we flip the board for the second player, we also need to flip the mask
        states = torch.stack(
            [
                torch.tensor(state.grid)
                if state.player == Player.FIRST
                else torch.tensor(state.grid).t()
                for state in states
            ]
        )
        legal_moves = (states == CellState.EMPTY).to(
            dtype=torch.float32, device=self.device
        )
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    # Since we flip the board for the second player, we need to flip it back
    def forward(self, states: Sequence[HexState]) -> Tuple[torch.Tensor, torch.Tensor]:
        value, action_probabilities = super().forward(states)
        action_probabilities = torch.where(
            torch.tensor([state.player == 0 for state in states]).reshape((-1, 1, 1)),
            torch.transpose(action_probabilities, 1, 2),
            action_probabilities,
        )
        return value, action_probabilities

    def states_to_tensor(self, states: Sequence[HexState]) -> torch.Tensor:
        players = torch.stack(
            [
                torch.full((self.grid_size, self.grid_size), fill_value=state.player)
                for state in states
            ]
        )
        # We want everything to be from the perspective of the current player.
        # We also want a consistent orientation, the current player's goal
        # should always be connecting north-south. This means we need to flip
        # the board for the second player.
        grids = torch.stack(
            [
                torch.tensor(state.grid)
                if state.player == Player.FIRST
                else torch.tensor(state.grid).t()
                for state in states
            ]
        )
        current_player = (
            grids
            == torch.tensor([state.player for state in states]).reshape((-1, 1, 1))
        ).float()
        other_player = (
            grids
            == torch.tensor([state.player.opposite() for state in states]).reshape(
                (-1, 1, 1)
            )
        ).float()
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.stack((current_player, other_player, players), dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(
        self, states: Sequence[HexState], distributions: Sequence[Sequence[float]]
    ) -> torch.Tensor:
        targets = torch.tensor(distributions, dtype=torch.float32)
        second_player_states = [state.player == Player.SECOND for state in states]
        targets[second_player_states, :] = (
            targets[second_player_states, :]
            .reshape((self.grid_size, self.grid_size))
            .t()
            .reshape((-1,))
        )
        assert targets.shape == (len(distributions), self.grid_size ** 2)
        return targets

    def copy(self) -> "ConvolutionalHexNet":
        net = ConvolutionalHexNet(
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
