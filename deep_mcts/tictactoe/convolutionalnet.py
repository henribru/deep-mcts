import random
from typing import Tuple, Dict, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from deep_mcts.convolutionalnet import (
    ConvolutionalBlock,
    ResidualBlock,
    PolicyHead,
    ValueHead,
    ConvolutionalNet,
)
from deep_mcts.game import CellState
from deep_mcts.gamenet import GameNet, DEVICE
from deep_mcts.tictactoe.game import TicTacToeAction, TicTacToeState, TicTacToeManager


class ConvolutionalTicTacToeNet(GameNet[TicTacToeState, TicTacToeAction]):
    manager: TicTacToeManager

    def __init__(self) -> None:
        super().__init__()
        self.net = ConvolutionalNet(
            num_residual=3,
            grid_size=3,
            in_channels=3,
            channels=16,
            policy_features=9,
            policy_shape=(3, 3),
        )
        self.manager = TicTacToeManager()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

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

    def sampling_policy(self, state: TicTacToeState) -> TicTacToeAction:
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :, :]
        assert action_probabilities.shape == (3, 3)
        action = np.random.choice(3 ** 2, p=action_probabilities.flatten())
        y, x = np.unravel_index(action, action_probabilities.shape)
        action = TicTacToeAction((x, y))
        assert action in self.manager.legal_actions(state)
        return action

    def greedy_policy(
        self, state: TicTacToeState, epsilon: float = 0
    ) -> TicTacToeAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.manager.legal_actions(state))
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :, :]
        assert action_probabilities.shape == (3, 3)
        y, x = np.unravel_index(
            torch.argmax(action_probabilities), action_probabilities.shape
        )
        assert action_probabilities[y][x] == action_probabilities.max()
        action = TicTacToeAction((x, y))
        assert action in self.manager.legal_actions(state)
        return action

    def evaluate_state(
        self, state: TicTacToeState
    ) -> Tuple[float, Dict[TicTacToeAction, float]]:
        value, probabilities = self.forward(state)
        actions = {
            TicTacToeAction((x, y)): probabilities[0, y, x].item()
            for y in range(3)
            for x in range(3)
        }
        legal_actions = set(self.manager.legal_actions(state))
        assert all(
            action in legal_actions
            for action, probability in actions.items()
            if probability != 0
        )
        return value, actions

    def state_to_tensor(self, state: TicTacToeState) -> torch.Tensor:
        return self.states_to_tensor([state])

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
        distributions: Sequence[Mapping[TicTacToeAction, float]],
    ) -> torch.Tensor:
        targets = torch.zeros(len(distributions), 3, 3).float()
        for i, distribution in enumerate(distributions):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y][x] = probability
        assert targets.shape == (len(distributions), 3, 3)
        return targets

    def copy(self) -> "ConvolutionalTicTacToeNet":
        net = ConvolutionalTicTacToeNet()
        net.net.load_state_dict(self.net.state_dict())
        return net
