from __future__ import annotations
import random
from typing import List, Tuple, Dict

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from anet import ANET, cross_entropy, DEVICE
from hex import HexAction, HexState, HexStateManager


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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.out = nn.Linear(in_features=grid_size ** 2, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.out(x.reshape(-1, self.grid_size ** 2))
        return x


class ConvolutionalHexNet(nn.Module):
    def __init__(self, num_residual: int, grid_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.residual_blocks = [
        #     ResidualBlock(in_channels=1, out_channels=1, kernel_size=3)
        #     for _ in range(num_residual)
        # ]
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        self.policy_head = PolicyHead()
        self.value_head = ValueHead(grid_size)

    def forward(self, x):
        input = x
        assert input.shape == (input.shape[0], 3, input.shape[2], input.shape[3])
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # for residual_block in self.residual_blocks:
        #     x = residual_block(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        value, probabilities = self.value_head(x), self.policy_head(x)
        assert probabilities.shape == (input.shape[0], 1, input.shape[2], input.shape[3])
        assert value.shape == (input.shape[0], 1)
        return value, probabilities


class ConvolutionalHexANET(ANET):
    grid_size: int
    state_manager: HexStateManager

    def __init__(self, grid_size: int):
        self.net = ConvolutionalHexNet(1, grid_size).to(DEVICE)
        self.grid_size = grid_size
        self.policy_criterion = cross_entropy
        self.value_criterion = nn.MSELoss().to(DEVICE)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.state_manager = HexStateManager(grid_size)

    def mask_illegal_moves(self, states: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        legal_moves = (
            states[:, 0, np.newaxis, ...] | states[:, 1, np.newaxis, ...]
        ) == 0
        assert legal_moves.shape == output.shape
        result = output * legal_moves.float()
        assert result.shape == output.shape
        return result

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
        return HexAction((x, y))

    def greedy_policy(self, state: HexState, epsilon=0) -> HexAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.state_manager.legal_actions(state))
        _, action_probabilities = self.forward(state)
        action_probabilities = action_probabilities[
            0, 0, :, :
        ]
        assert action_probabilities.shape == (self.grid_size, self.grid_size)
        y, x = np.unravel_index(
            np.argmax(action_probabilities), action_probabilities.shape
        )
        return HexAction((x, y))

    def evaluate_state(self, state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        value, probabilities = self.forward(state)
        actions = {}
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                actions[HexAction((x, y))] = probabilities[0, 0, y, x]
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
        states = np.stack([state.grid for state in states], axis=0)
        first_player = states == 0
        second_player = states == 1
        #  assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        states = np.stack((first_player, second_player, players), axis=1)
        tensor = torch.as_tensor(states, device=DEVICE)
        assert tensor.shape == (len(states), 3, self.grid_size, self.grid_size)
        return tensor

    def distributions_to_tensor(self, distributions: List[Dict[HexAction, float]]) -> torch.Tensor:
        targets = np.zeros(
            (len(distributions), self.grid_size, self.grid_size), dtype=np.float32
        )
        for i, distribution in enumerate(distributions):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y][x] = probability
        targets = torch.as_tensor(targets, device=DEVICE)
        assert targets.shape == (len(distributions), self.grid_size, self.grid_size)
        return targets
