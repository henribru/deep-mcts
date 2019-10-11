import random
from typing import Tuple, Dict, Mapping, Sequence, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from deep_mcts.gamenet import GameNet, DEVICE
from deep_mcts.tictactoe.game import TicTacToeAction, TicTacToeState, TicTacToeManager


class ConvolutionalBlock(nn.Module):  # type: ignore
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):  # type: ignore
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = x
        assert input.shape == (
            input.shape[0],
            self.in_channels,
            input.shape[2],
            input.shape[3],
        )
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.projection(input)
        x = F.relu(x)
        assert x.shape == (
            input.shape[0],
            self.out_channels,
            input.shape[2],
            input.shape[3],
        )
        return x


class PolicyHead(nn.Module):  # type: ignore
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ValueHead(nn.Module):  # type: ignore
    def __init__(self, grid_size: int, in_channels: int, hidden_units: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )
        self.fc1 = nn.Linear(in_features=grid_size ** 2, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv1(x)
        x = self.fc1(x.reshape((x.shape[0], -1)))
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ConvolutionalTicTacToeModule(nn.Module):  # type: ignore
    def __init__(self, num_residual: int, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalBlock(
            in_channels=3, out_channels=channels, kernel_size=3, padding=1
        )
        self.residual_blocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(num_residual)
            ]
        )
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(3, channels, hidden_units=64)

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = x
        assert input.shape == (input.shape[0], 3, input.shape[2], input.shape[3])
        x = self.conv1(x)
        for residual_block in self.residual_blocks:  # type: ignore
            x = residual_block(x)
        value, probabilities = self.value_head(x), self.policy_head(x)
        probabilities = probabilities.squeeze(1)
        assert probabilities.shape == (input.shape[0], input.shape[2], input.shape[3])
        assert value.shape == (input.shape[0], 1)
        return value, probabilities


class ConvolutionalTicTacToeNet(GameNet[TicTacToeState, TicTacToeAction]):
    manager: TicTacToeManager

    def __init__(self) -> None:
        super().__init__()
        self.net = ConvolutionalTicTacToeModule(num_residual=3, channels=16).to(DEVICE)
        self.manager = TicTacToeManager()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

    def _mask_illegal_moves(
        self, states: Sequence[TicTacToeState], output: torch.Tensor
    ) -> torch.Tensor:
        states = np.stack([state.grid for state in states], axis=0)
        legal_moves = torch.as_tensor(states == -1, dtype=torch.float32, device=DEVICE)
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    def sampling_policy(self, state: TicTacToeState) -> TicTacToeAction:
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[0, :, :]
        assert action_probabilities.shape == (3, 3)
        action = np.random.choice(
            action_probabilities.size, p=action_probabilities.flatten()
        )
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
            np.argmax(action_probabilities), action_probabilities.shape
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
            TicTacToeAction((x, y)): probabilities[0, y, x]
            for y in range(3)
            for x in range(3)
        }
        assert set(
            action for action, probability in actions.items() if probability != 0
        ) == set(self.manager.legal_actions(state))
        return value, actions

    def state_to_tensor(self, state: TicTacToeState) -> torch.Tensor:
        return self.states_to_tensor([state])

    def states_to_tensor(self, states: Sequence[TicTacToeState]) -> torch.Tensor:
        players = np.array(
            [np.full(shape=(3, 3), fill_value=state.player) for state in states]
        )
        # We want everything to be from the perspective of the current player.
        grids = np.stack([state.grid for state in states], axis=0)
        current_player = grids == np.array([state.player for state in states]).reshape(
            (-1, 1, 1)
        )
        other_player = grids == np.array(
            [0 if state.player == 1 else 1 for state in states]
        ).reshape((-1, 1, 1))
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        tensor = torch.as_tensor(
            np.stack((current_player, other_player, players), axis=1)
        )
        assert tensor.shape == (len(states), 3, 3, 3)
        return tensor

    def distributions_to_tensor(
        self,
        states: Sequence[TicTacToeState],
        distributions: Sequence[Mapping[TicTacToeAction, float]],
    ) -> torch.Tensor:
        targets = np.zeros((len(distributions), 3, 3), dtype=np.float32)
        for i, distribution in enumerate(distributions):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y][x] = probability
        targets = torch.as_tensor(targets)
        assert targets.shape == (len(distributions), 3, 3)
        return targets

    def copy(self) -> "ConvolutionalTicTacToeNet":
        net = ConvolutionalTicTacToeNet()
        net.net.load_state_dict(self.net.state_dict())
        return net
