from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, TypeVar, Sequence, Generic, Mapping, Type, Any

import numpy as np
import torch.optim.optimizer
import torch.nn as nn
import torch.nn.functional as F

from deep_mcts.mcts import Action, State

from deep_mcts.tournament import Agent

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cpu")


def cross_entropy(
    pred: torch.Tensor, soft_targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    pred = pred.reshape((pred.shape[0], -1))
    soft_targets = soft_targets.reshape((soft_targets.shape[0], -1))
    result = torch.sum(-soft_targets * F.log_softmax(pred, dim=1), dim=1)
    assert result.shape == (pred.shape[0],)
    if reduction == "mean":
        result = torch.mean(result)
        assert result.shape == ()
    elif reduction == "sum":
        result = torch.sum(result)
        assert result.shape == ()
    return result


_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)
_T = TypeVar("_T", bound="GameNet")  # type: ignore


class GameNet(ABC, Generic[_S, _A]):
    net: "nn.Module[Any]"
    policy_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    value_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: "torch.optim.optimizer.Optimizer"

    def __init__(self) -> None:
        self.policy_criterion = cross_entropy  # type: ignore
        self.value_criterion = nn.MSELoss().to(DEVICE)  # type: ignore

    def forward(self, state: _S) -> Tuple[float, np.ndarray]:
        states = self.state_to_tensor(state).to(DEVICE)
        with torch.autograd.no_grad():
        value, probabilities = self.net.forward(states.float())
        value = torch.tanh(value)
        # The output value is from the perspective of the current player,
        # but MCTS expects it to be independent of the player
        if state.player == 1:
            value = -value
        shape = probabilities.shape
        assert probabilities.shape == shape
        probabilities = F.softmax(probabilities.reshape((1, -1)), dim=1).reshape(shape)
        assert probabilities.shape == shape
        probabilities = self._mask_illegal_moves([state], probabilities)
        assert probabilities.shape == shape
        probabilities = probabilities / torch.sum(
            probabilities, dim=tuple(range(1, probabilities.dim())), keepdim=True
        )
        assert probabilities.shape == shape
        assert torch.allclose(
            torch.sum(probabilities, dim=tuple(range(1, probabilities.dim()))),
            torch.tensor([1.0], device=DEVICE),
        )
        return value.item(), probabilities.cpu().detach().numpy()

    @abstractmethod
    def _mask_illegal_moves(
        self, states: Sequence[_S], output: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def sampling_policy(self, state: _S) -> _A:
        ...

    @abstractmethod
    def greedy_policy(self, state: _S, epsilon: float = 0) -> _A:
        ...

    @abstractmethod
    def evaluate_state(self, state: _S) -> Tuple[float, Dict[_A, float]]:
        ...

    def train(
        self,
        states: torch.Tensor,
        probability_targets: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> None:
        values, probabilities = self.net.forward(states.float())
        values = torch.tanh(values)
        assert probabilities.shape[0] == states.shape[0]
        assert values.shape == (states.shape[0], 1)
        # shape = output.shape
        # output = F.softmax(output.reshape((shape[0], -1)), dim=1).reshape(shape)
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
        assert probabilities.shape == probability_targets.shape
        assert values.shape == value_targets.shape
        loss = self.policy_criterion(  # type: ignore
            probabilities, probability_targets
        ) + self.value_criterion(  # type: ignore
            values, value_targets
        )
        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path))

    @abstractmethod
    def state_to_tensor(self, state: _S) -> torch.Tensor:
        ...

    @abstractmethod
    def states_to_tensor(self, states: Sequence[_S]) -> torch.Tensor:
        ...

    @abstractmethod
    def distributions_to_tensor(
        self, states: Sequence[_S], distributions: Sequence[Mapping[_A, float]]
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def copy(self: _T) -> _T:
        ...

    @classmethod
    def from_path(cls: Type[_T], path: str, *args: Any, **kwargs: Any) -> _T:
        anet = cls(*args, **kwargs)  # type: ignore
        anet.load(path)
        return anet


class GameNetAgent(Agent[_S, _A], ABC):
    net: GameNet[_S, _A]

    def __init__(self, net: GameNet[_S, _A]):
        self.net = net

    def reset(self) -> None:
        pass


class GreedyGameNetAgent(GameNetAgent[_S, _A]):
    epsilon: float

    def __init__(self, net: GameNet[_S, _A], epsilon: float = 0):
        super().__init__(net)
        self.epsilon = epsilon

    def play(self, state: _S) -> _A:
        return self.net.greedy_policy(state, self.epsilon)


class SamplingGameNetAgent(GameNetAgent[_S, _A]):
    def play(self, state: _S) -> _A:
        return self.net.sampling_policy(state)
