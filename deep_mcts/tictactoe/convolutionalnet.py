from typing import Tuple, Mapping, Sequence, Type, Any, Optional, Dict

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
    value_head_hidden_units: int

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
        value_head_hidden_units: int = 128,
    ) -> None:
        super().__init__(
            ConvolutionalNet(
                num_residual=num_residual,
                grid_size=3,
                in_channels=3,
                channels=channels,
                value_head_hidden_units=value_head_hidden_units,
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
        self.value_head_hidden_units = value_head_hidden_units

    def states_to_tensor(self, states: Sequence[TicTacToeState]) -> torch.Tensor:
        players, opposite_players, grids = zip(
            *[(state.player, state.player.opposite(), state.grid) for state in states]
        )
        players = torch.tensor(players).reshape(-1, 1, 1)
        opposite_players = torch.tensor(opposite_players).reshape(-1, 1, 1)

        player_grids = torch.full((len(states), 3, 3), fill_value=-1.0)
        player_grids[:] = players

        # We want everything to be from the perspective of the current player.
        grids = torch.tensor(grids)
        current_player = (grids == players).float()
        other_player = (grids == opposite_players).float()
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.stack([current_player, other_player, player_grids], dim=1)
        assert tensor.shape == (len(states), 3, 3, 3)
        return tensor

    def copy(self) -> "ConvolutionalTicTacToeNet":
        net = ConvolutionalTicTacToeNet(
            self.manager,
            self.optimizer_cls,
            self.optimizer_args,
            self.optimizer_kwargs,
            self.num_residual,
            self.channels,
            self.value_head_hidden_units,
        )
        net.load_state_dict(self.net.state_dict())
        return net

    @classmethod
    def from_path_full(
        cls, path: str, manager: Optional[GameManager[TicTacToeState]] = None,
    ) -> "ConvolutionalTicTacToeNet":
        parameters = torch.load(path, map_location=torch.device("cpu"))
        state_dict = parameters.pop("state_dict")
        net = cls(
            manager=manager,
            optimizer_cls=parameters["optimizer_cls"],
            optimizer_args=parameters["optimizer_args"],
            optimizer_kwargs=parameters["optimizer_kwargs"],
            num_residual=parameters["num_residual"],
            channels=parameters["channels"],
            value_head_hidden_units=parameters["value_head_hidden_units"],
        )
        net.load_state_dict(parameters["state_dict"])
        return net

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            "num_residual": self.num_residual,
            "channels": self.channels,
            "value_head_hidden_units": self.value_head_hidden_units,
        }
