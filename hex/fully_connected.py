from __future__ import annotations

import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from anet import ANET, cross_entropy, DEVICE
from hex.game import HexAction, HexState, HexManager


class FullyConnectedHexNet(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.fc1 = nn.Linear(grid_size ** 2 * 2 + 1, 128)
        self.fc2 = nn.Linear(128, grid_size ** 2)

    def forward(self, x):
        input = x
        assert input.shape == (input.shape[0], self.grid_size ** 2 * 2 + 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        assert x.shape == (input.shape[0], (input.shape[1] - 1) / 2)
        return x


class FullyConnectedHexANET(ANET[HexState, HexAction]):
    grid_size: int
    state_manager: HexManager

    def __init__(self, grid_size: int):
        self.net = FullyConnectedHexNet(grid_size)
        self.grid_size = grid_size
        self.policy_criterion = cross_entropy
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.state_manager = HexManager(grid_size)

    def mask_illegal_moves(self, states: torch.Tensor, output: torch.Tensor):
        legal_moves = (
            states[:, : self.grid_size ** 2] | states[:, self.grid_size ** 2 : -1]
        ) == 0
        result = output * legal_moves.float()
        assert result.shape == output.shape
        return result

    def sampling_policy(self, state: HexState) -> HexAction:
        action_probabilities = self.forward(state)[0, :]
        assert action_probabilities.shape == (self.grid_size ** 2,)
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        x, y = action % self.grid_size, action // self.grid_size
        return HexAction((x, y))

    def greedy_policy(self, state: HexState, epsilon=0) -> HexAction:
        if epsilon > 0:
            p = random.random()
            if p < epsilon:
                return random.choice(self.state_manager.legal_actions(state))
        action_probabilities = self.forward(state)[0, :]
        assert action_probabilities.shape == (self.grid_size ** 2,)
        action = np.argmax(action_probabilities)
        x, y = action % self.grid_size, action // self.grid_size
        return HexAction((x, y))

    def evaluate_state(self, state: HexState) -> Tuple[float, Dict[HexAction, float]]:
        value, probabilities = self.forward(state)[0, :]
        assert probabilities.shape == (self.grid_size ** 2,)
        actions = {}
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                actions[HexAction((x, y))] = probabilities[y * self.grid_size + x]
        return value, actions

    def state_to_tensor(self, state: HexState):
        grid = state.grid
        tensor = [0] * (self.grid_size ** 2 * 2 + 1)
        tensor[-1] = state.player
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                tensor[2 * (y * self.grid_size + x)] = 1 if grid[y][x] == 0 else 0
                tensor[2 * (y * self.grid_size + x) + 1] = 1 if grid[y][x] == 1 else 0
        tensor = torch.tensor(tensor)
        assert tensor.shape == (2 * self.grid_size ** 2 + 1)
        return tensor

    def states_to_tensor(self, states: List[HexState]):
        players = np.array([state.player for state in states]).reshape((len(states), -1))
        old_states = states
        states = np.stack([state.grid for state in states]).reshape((len(states), -1))
        for i in range(len(old_states)):
            assert np.all(states[i, :] == np.array(old_states[i].grid).flatten())
        first_player = states == 0
        second_player = states == 1
        assert np.all((first_player.sum(axis=1) - second_player.sum(axis=1)) <= 1)
        states = np.concatenate((first_player, second_player, players), axis=1)
        tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        assert tensor.shape == (len(states), 2 * self.grid_size ** 2 + 1)
        return tensor

    def distributions_to_tensor(self, distributions: List[Dict[HexAction, float]]):
        targets = np.zeros((len(distributions), self.grid_size ** 2), dtype=np.float32)
        for i, distribution in enumerate(distributions):
            for action, probability in distribution.items():
                x, y = action.coordinate
                targets[i][y * self.grid_size + x] = probability
        targets = torch.as_tensor(targets, device=DEVICE)
        assert targets.shape == (len(distributions), self.grid_size ** 2)
        return targets
