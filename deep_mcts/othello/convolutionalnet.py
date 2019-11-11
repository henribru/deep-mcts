from typing import Tuple, Dict, Mapping, Sequence, Type, Any, Optional

import torch
import torch.optim
import torch.optim.optimizer

from deep_mcts.game import GameManager
from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.gamenet import GameNet
from deep_mcts.othello.game import (
    OthelloPass,
    OthelloMove,
    OthelloAction,
    OthelloState,
    OthelloManager,
)


class ConvolutionalOthelloNet(GameNet[OthelloState, OthelloAction]):
    grid_size: int
    num_residual: int
    channels: int

    def __init__(
        self,
        grid_size: int,
        manager: Optional[GameManager[OthelloState, OthelloAction]] = None,
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"] = torch.optim.SGD,
        optimizer_args: Tuple[Any, ...] = (),
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.01, "momentum": 0.9},
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

    def _mask_illegal_moves(
        self, states: Sequence[OthelloState], output: torch.Tensor
    ) -> torch.Tensor:
        legal_moves = torch.zeros(len(states), self.grid_size ** 2 + 1)
        for i, state in enumerate(states):
            for action in self.manager.legal_actions(state):
                if isinstance(action, OthelloPass):
                    legal_moves[i][-1] = 1.0
                else:
                    x, y = action.coordinate
                    legal_moves[i][y * self.grid_size + x] = 1.0
        assert legal_moves.shape == output.shape
        result = output * legal_moves.to(self.device)
        assert result.shape == output.shape
        return result

    def sampling_policy(self, state: OthelloState) -> OthelloAction:
        raise NotImplementedError

    def greedy_policy(self, state: OthelloState, epsilon: float = 0) -> OthelloAction:
        raise NotImplementedError

    def evaluate_state(
        self, state: OthelloState
    ) -> Tuple[float, Dict[OthelloAction, float]]:
        value, probabilities = self.forward(state)
        actions: Dict[OthelloAction, float] = {
            OthelloMove((x, y)): probabilities[0, y * self.grid_size + x].item()
            for y in range(self.grid_size)
            for x in range(self.grid_size)
        }
        actions[OthelloPass()] = probabilities[0, -1].item()
        if __debug__:
            legal_actions = set(self.manager.legal_actions(state))
            assert all(
                action in legal_actions
                for action, probability in actions.items()
                if probability != 0
            )
        return value, actions

    def state_to_tensor(self, state: OthelloState) -> torch.Tensor:
        return self.states_to_tensor([state])

    def states_to_tensor(self, states: Sequence[OthelloState]) -> torch.Tensor:
        players = torch.stack(
            [
                torch.full((self.grid_size, self.grid_size), fill_value=state.player)
                for state in states
            ]
        )
        # We want everything to be from the perspective of the current player.
        grids = torch.stack([torch.tensor(state.grid) for state in states])
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
        tensor = torch.stack([current_player, other_player, players], dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[OthelloState],
        distributions: Sequence[Mapping[OthelloAction, float]],
    ) -> torch.Tensor:
        targets = torch.zeros(len(distributions), self.grid_size ** 2 + 1).float()
        for i, (state, distribution) in enumerate(zip(states, distributions)):
            for action, probability in distribution.items():
                if isinstance(action, OthelloPass):
                    targets[i][-1] = probability
                else:
                    x, y = action.coordinate
                    targets[i][y * self.grid_size + x] = probability
        assert targets.shape == (len(distributions), self.grid_size ** 2 + 1)
        return targets

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
