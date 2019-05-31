from __future__ import annotations

import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from deep_mcts.gamenet import GameNet, cross_entropy, DEVICE
from deep_mcts.hex.game import HexAction, HexState, HexManager


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
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


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, grid_size: int, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=grid_size ** 2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.fc1(x.reshape((x.shape[0], -1)))
        x = F.relu(x)
        x = self.fc2(x)
        return x


class ConvolutionalHexModule(nn.Module):
    def __init__(self, num_residual: int, grid_size: int, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.residual_blocks = [
            ResidualBlock(in_channels=channels, out_channels=channels, kernel_size=3)
            for _ in range(num_residual)
        ]
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(grid_size, channels)

    def forward(self, x):
        input = x
        assert input.shape == (input.shape[0], 3, input.shape[2], input.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        value, probabilities = self.value_head(x), self.policy_head(x)
        assert probabilities.shape == (input.shape[0], 1, input.shape[2], input.shape[3])
        assert value.shape == (input.shape[0], 1)
        return value, probabilities

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        res.residual_blocks = [residual_block.to(*args, **kwargs) for residual_block in res.residual_blocks]
        return res


class ConvolutionalHexNet(GameNet[HexState, HexAction]):
    grid_size: int
    hex_manager: HexManager

    def __init__(self, grid_size: int):
        self.net = ConvolutionalHexModule(num_residual=3, grid_size=grid_size, channels=16).to(DEVICE)
        self.grid_size = grid_size
        self.policy_criterion = cross_entropy
        self.value_criterion = nn.MSELoss().to(DEVICE)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.hex_manager = HexManager(grid_size)

    def mask_illegal_moves(self, states: List[HexState], output: torch.Tensor) -> torch.Tensor:
        states = np.stack([state.grid if state.player == 1 else np.transpose(state.grid) for state in states], axis=0)
        states = states[:, np.newaxis, ...]
        legal_moves = torch.as_tensor(states == -1, dtype=torch.float32)
        assert legal_moves.shape == output.shape
        result = output * legal_moves
        assert result.shape == output.shape
        return result

    # Since we flip the board for player 0, we need to flip it back
    def forward(self, state: HexState) -> Tuple[float, np.ndarray]:
        value, action_probabilities = super().forward(state)
        if state.player == 0:
            action_probabilities = np.swapaxes(action_probabilities, 2, 3)
        return value, action_probabilities

    def sampling_policy(self, state: HexState) -> HexAction:
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[
            0, 0, :, :
        ]
        assert action_probabilities.shape == (self.grid_size, self.grid_size)
        action = np.random.choice(
            action_probabilities.size, p=action_probabilities.flatten()
        )
        y, x = np.unravel_index(action, action_probabilities.shape)
        action = HexAction((x, y))
        assert action in self.hex_manager.legal_actions(state)
        return action

    def greedy_policy(self, state: HexState, epsilon=0) -> HexAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.hex_manager.legal_actions(state))
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[
            0, 0, :, :
        ]
        assert action_probabilities.shape == (self.grid_size, self.grid_size)
        y, x = np.unravel_index(
            np.argmax(action_probabilities), action_probabilities.shape
        )
        action = HexAction((x, y))
        assert action in self.hex_manager.legal_actions(state)
        return action

    def evaluate_state(self, state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        value, probabilities = self.forward(state)
        actions = {}
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                actions[HexAction((x, y))] = probabilities[0, 0, y, x]
        assert set(actions.keys()) == set(self.hex_manager.legal_actions())
        return value, actions

    def state_to_tensor(self, state: HexState) -> torch.Tensor:
        return self.states_to_tensor([state])

    def states_to_tensor(self, states: List[HexState]) -> torch.Tensor:
        players = np.array(
            [
                np.full(shape=(self.grid_size, self.grid_size), fill_value=state.player)
                for state in states
            ]
        )
        # We want everything to be from the perspective of the current player.
        # We also want a consistent orientation, the current player's goal
        # should always be connecting north-south. This means we need to flip
        # the board for player 0.
        grids = np.stack([state.grid if state.player == 1 else np.transpose(state.grid) for state in states], axis=0)
        current_player = grids == np.array([state.player for state in states]).reshape((-1, 1, 1))
        other_player = grids == np.array([0 if state.player == 1 else 1 for state in states]).reshape((-1, 1, 1))
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        states = np.stack((current_player, other_player, players), axis=1)
        tensor = torch.as_tensor(states, device=DEVICE)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(self, states: List[HexState], distributions: List[Dict[HexAction, float]]) -> torch.Tensor:
        targets = np.zeros(
            (len(distributions), self.grid_size, self.grid_size), dtype=np.float32
        )
        for i, (state, distribution) in enumerate(zip(states, distributions)):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y][x] = probability
            # Since we flip the board for player 0, we also need to flip the targets
            if state.player == 0:
                targets[i] = targets[i].T
        targets = torch.as_tensor(targets, device=DEVICE)
        assert targets.shape == (len(distributions), self.grid_size, self.grid_size)
        return targets
