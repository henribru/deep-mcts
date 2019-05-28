import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, TypeVar, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcts import Action, State

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda")


def cross_entropy(pred, soft_targets, reduction='mean'):
    pred = pred.reshape(pred.shape[0], -1)
    soft_targets = soft_targets.reshape(soft_targets.shape[0], -1)
    result = torch.sum(-soft_targets * F.log_softmax(pred, dim=1), dim=1)
    assert result.shape == (pred.shape[0],)
    if reduction == 'mean':
        result = torch.mean(result)
        assert result.shape == ()
    elif reduction == 'sum':
        result = torch.sum(result)
        assert result.shape == ()
    return result


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)


class ANET(ABC):
    net: nn.Module
    policy_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    value_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer

    def forward(self, state: S) -> Tuple[float, np.ndarray]:
        states = self.states_to_tensor([state])
        value, probabilities = self.net.forward(states.float())
        shape = probabilities.shape
        assert probabilities.shape == shape
        probabilities = self.mask_illegal_moves(states, probabilities)
        assert probabilities.shape == shape
        probabilities = probabilities / torch.sum(
            probabilities, dim=tuple(range(1, probabilities.dim())), keepdim=True
        )
        assert probabilities.shape == shape
        assert torch.allclose(
            torch.sum(probabilities, dim=tuple(range(1, probabilities.dim()))), torch.tensor([1.0])
        )
        return value.item(), probabilities.cpu().detach().numpy()

    @abstractmethod
    def mask_illegal_moves(self, states: torch.Tensor, output: torch.Tensor):
        ...

    @abstractmethod
    def sampling_policy(self, state: S) -> A:
        ...

    @abstractmethod
    def greedy_policy(self, state: S, epsilon=0) -> A:
        ...

    @abstractmethod
    def evaluate_state(self, state: S) -> Tuple[float, Dict[Action, float]]:
        ...

    def train(self, replay_buffer: Sequence[Tuple[S, Dict[A, float], float]]) -> None:
        self.optimizer.zero_grad()
        examples = random.sample(replay_buffer, min(512, len(replay_buffer)))
        states, probability_targets, value_targets = zip(*examples)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        assert value_targets.shape[0] == len(examples)
        probability_targets = self.distributions_to_tensor(probability_targets)
        assert probability_targets.shape[0] == len(examples)
        states = self.states_to_tensor(states)
        assert states.shape[0] == len(examples)
        values, probabilities = self.net.forward(states.float())
        values = torch.tanh(values)
        assert probabilities.shape[0] == states.shape[0]
        assert values.shape == (states.shape[0], 1)
        # shape = output.shape
        # output = F.softmax(output.reshape(shape[0], -1), dim=1).reshape(shape)
        # assert torch.allclose(
        #     torch.sum(output, dim=tuple(range(1, output.dim()))), torch.Tensor([1.0])
        # )
        # assert output.shape == shape
        #  output = self.mask_illegal_moves(states, output)
        #  assert output.shape == shape
        #  output = output / torch.sum(
        #      output, dim=tuple(range(1, output.dim())), keepdim=True
        #  )
        #  assert output.shape == shape
        #  assert torch.allclose(
        #      torch.sum(output, dim=tuple(range(1, output.dim()))), torch.Tensor([1.0])
        #  )
        loss = self.policy_criterion(probabilities, probability_targets) + self.value_criterion(values, value_targets)
        assert loss.shape == ()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))

    @abstractmethod
    def state_to_tensor(self, state: S) -> torch.Tensor:
        ...

    @abstractmethod
    def states_to_tensor(self, states: List[S]) -> torch.Tensor:
        ...

    @abstractmethod
    def distributions_to_tensor(
        self, distributions: List[Dict[A, float]]
    ) -> torch.Tensor:
        ...

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        anet = cls(*args, **kwargs)
        anet.net.load_state_dict(torch.load(path))
        return anet
