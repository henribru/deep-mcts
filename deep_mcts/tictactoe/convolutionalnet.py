from typing import Tuple, Mapping, Sequence, Type, Any, Optional

import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import CellState, GameManager
from deep_mcts.gamenet import GameNet
from deep_mcts.tictactoe.game import Action, TicTacToeState, TicTacToeManager


class ConvolutionalTicTacToeNet(GameNet[TicTacToeState]):
    num_residual: int
    channels: int

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
        num_residual: int = 3,
        channels: int = 3,
    ) -> None:
        super().__init__(
            ConvolutionalNet(
                num_residual=num_residual,
                grid_size=3,
                in_channels=3,
                channels=channels,
                policy_features=9,
                policy_shape=(3, 3),
            ),
            manager if manager is not None else TicTacToeManager(),
            optimizer_cls,
            optimizer_args,
            optimizer_kwargs,
        )
        self.num_residual = num_residual
        self.channels = channels

    def _mask_illegal_moves(
        self, states: Sequence[TicTacToeState], output: torch.Tensor
    ) -> torch.Tensor:
        states = torch.tensor([state.grid for state in states])
        legal_moves = (states == CellState.EMPTY).to(
            dtype=torch.float32, device=self.device
        )
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    def states_to_tensor(self, states: Sequence[TicTacToeState]) -> torch.Tensor:
        players = torch.stack(
            [torch.full((3, 3), fill_value=state.player) for state in states]
        )
        # We want everything to be from the perspective of the current player.
        grids = torch.tensor([state.grid for state in states])
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
        assert tensor.shape == (len(states), 3, 3, 3)
        return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[TicTacToeState],
        distributions: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        targets = torch.tensor(distributions, dtype=torch.float32)
        assert targets.shape == (len(distributions), 3, 3)
        return targets

    def copy(self) -> "ConvolutionalTicTacToeNet":
        net = ConvolutionalTicTacToeNet(
            self.manager,
            self.optimizer_cls,
            self.optimizer_args,
            self.optimizer_kwargs,
            self.num_residual,
            self.channels,
        )
        net.net.load_state_dict(self.net.state_dict())
        return net
