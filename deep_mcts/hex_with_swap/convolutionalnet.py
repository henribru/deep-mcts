from typing import Tuple, Dict, Mapping, Sequence

import torch
import torch.optim

from deep_mcts.convolutionalnet import ConvolutionalNet
from deep_mcts.game import CellState, Player
from deep_mcts.gamenet import GameNet
from deep_mcts.hex_with_swap.game import (
    HexWithSwapManager,
    HexState,
    HexWithSwapAction,
    HexAction,
    HexSwap,
)


class ConvolutionalHexWithSwapNet(GameNet[HexState, HexWithSwapAction]):
    grid_size: int
    hex_manager: HexWithSwapManager

    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.net = ConvolutionalNet(
            num_residual=1,
            grid_size=grid_size,
            in_channels=3,
            channels=128,
            policy_features=grid_size ** 2 + 1,
            policy_shape=(grid_size ** 2 + 1,),
        )
        self.grid_size = grid_size
        self.hex_manager = HexWithSwapManager(grid_size)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

    def _mask_illegal_moves(
        self, states: Sequence[HexState], output: torch.Tensor
    ) -> torch.Tensor:
        states_ = states
        states = torch.stack(
            [
                torch.tensor(state.grid)
                if state.player == Player.FIRST
                else torch.tensor(state.grid).t()
                for state in states
            ]
        ).reshape((len(states), -1))
        legal_moves = (states == CellState.EMPTY).float()
        swaps = torch.tensor(
            [[HexSwap() in self.hex_manager.legal_actions(state)] for state in states_],
            dtype=torch.float32,
        )
        legal_moves = torch.cat([legal_moves, swaps], 1).to(self.device)
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    # Since we flip the board for player 0, we need to flip it back
    def forward(self, state: HexState) -> Tuple[float, torch.Tensor]:
        value, action_probabilities = super().forward(state)
        if state.player == Player.SECOND:
            action_probabilities[0, :-1] = (
                action_probabilities[0, :-1]
                .reshape((self.grid_size, self.grid_size))
                .t()
                .reshape((-1,))
            )
        return value, action_probabilities

    def sampling_policy(self, state: HexState) -> HexWithSwapAction:
        raise NotImplementedError

    def greedy_policy(self, state: HexState, epsilon: float = 0) -> HexWithSwapAction:
        raise NotImplementedError

    def evaluate_state(
        self, state: HexState
    ) -> Tuple[float, Dict[HexWithSwapAction, float]]:
        value, probabilities = self.forward(state)
        actions: Dict[HexWithSwapAction, float] = {
            HexAction((x, y)): probabilities[0, y * self.grid_size + x].item()
            for y in range(self.grid_size)
            for x in range(self.grid_size)
        }
        actions[HexSwap()] = probabilities[0, -1].item()
        legal_actions = set(self.hex_manager.legal_actions(state))
        assert all(
            action in legal_actions
            for action, probability in actions.items()
            if probability != 0
        )
        return value, actions

    def state_to_tensor(self, state: HexState) -> torch.Tensor:
        return self.states_to_tensor([state])

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
        # the board for player 0.
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
        tensor = torch.stack([current_player, other_player, players], dim=1)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[HexState],
        distributions: Sequence[Mapping[HexWithSwapAction, float]],
    ) -> torch.Tensor:
        targets = torch.zeros(
            len(distributions), self.grid_size * self.grid_size + 1
        ).float()
        for i, (state, distribution) in enumerate(zip(states, distributions)):
            for action, probability in distribution.items():
                if isinstance(action, HexSwap):
                    targets[i][-1] = probability
                else:
                    x, y = action.coordinate
                    targets[i][y * self.grid_size + x] = probability
            # Since we flip the board for player 0, we also need to flip the targets
            if state.player == Player.SECOND:
                targets[i, :-1] = (
                    targets[i, :-1]
                    .reshape((self.grid_size, self.grid_size))
                    .t()
                    .reshape((-1,))
                )
        assert targets.shape == (
            len(distributions),
            self.grid_size * self.grid_size + 1,
        )
        return targets

    def copy(self) -> "ConvolutionalHexWithSwapNet":
        net = ConvolutionalHexWithSwapNet(self.grid_size)
        net.net.load_state_dict(self.net.state_dict())
        return net
