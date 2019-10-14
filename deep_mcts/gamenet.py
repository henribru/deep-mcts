from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    TYPE_CHECKING,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.optimizer

from deep_mcts.game import Player, State, GameManager, Action
from deep_mcts.tournament import Agent

_S = TypeVar("_S", bound=State)
_T = TypeVar("_T", bound="GameNet")  # type: ignore[type-arg]

if TYPE_CHECKING:
    TensorPairModule = nn.Module[Tuple[torch.Tensor, torch.Tensor]]
else:
    TensorPairModule = nn.Module


class GameNet(ABC, Generic[_S]):
    net: TensorPairModule
    policy_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    value_criterion: nn.MSELoss
    optimizer: "torch.optim.optimizer.Optimizer"
    device: torch.device
    manager: GameManager[_S]

    def __init__(
        self,
        net: TensorPairModule,
        manager: GameManager[_S],
        optimizer_cls: Type["torch.optim.optimizer.Optimizer"],
        optimizer_args: Tuple[Any, ...],
        optimizer_kwargs: Mapping[str, Any],
    ) -> None:
        self.policy_criterion = cross_entropy  # type: ignore[assignment, misc]
        self.value_criterion = nn.MSELoss()
        self.device = torch.device("cpu")
        self.net = net
        self.manager = manager
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer_cls(  # type: ignore[call-arg]
            self.net.parameters(), *optimizer_args, **optimizer_kwargs
        )

    def forward(self, states: Sequence[_S]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.net.eval()
        states_ = states
        states = self.states_to_tensor(states).to(self.device)
        values: torch.Tensor
        with torch.autograd.no_grad():
            values, probabilities = self.net.forward(states.float())
        values = torch.tanh(values)
        # The output value is from the perspective of the current player,
        # but MCTS expects it to be independent of the player
        values[[state.player == Player.min_player() for state in states_], :] *= -1
        # MCTS uses a range of [0, 1]
        values = (values + 1) / 2
        shape = probabilities.shape
        assert probabilities.shape == shape
        probabilities = F.softmax(
            probabilities.reshape((len(states_), -1)), dim=1
        ).reshape(shape)
        assert probabilities.shape == shape
        probabilities = self._mask_illegal_moves(states_, probabilities)
        assert probabilities.shape == shape
        probabilities = probabilities / torch.sum(
            probabilities, dim=tuple(range(1, probabilities.dim())), keepdim=True
        )
        assert probabilities.shape == shape
        assert torch.allclose(
            torch.sum(probabilities, dim=tuple(range(1, probabilities.dim()))),
            torch.tensor([1.0], device=self.device),
        )
        values = values.flatten().cpu().detach()
        probabilities = probabilities.reshape((len(states_), -1)).cpu().detach()
        assert values.shape == (len(states_),)
        assert probabilities.shape[0] == len(states_)
        return values, probabilities

    def forward_single(self, state: _S) -> Tuple[float, Sequence[float]]:
        values, probabilities = self.forward([state])
        return values[0], probabilities[0]  # type: ignore[return-value]

    @abstractmethod
    def _mask_illegal_moves(
        self, states: Sequence[_S], output: torch.Tensor
    ) -> torch.Tensor:
        ...

    def train(
        self,
        states: torch.Tensor,
        probability_targets: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.net.train()
        values, probabilities = self.net.forward(states.float())
        values = torch.tanh(values)
        assert probabilities.shape[0] == states.shape[0]
        assert values.shape == (states.shape[0], 1)
        assert probabilities.shape == probability_targets.shape
        assert values.shape == value_targets.shape
        policy_loss = self.policy_criterion(  # type: ignore[misc, call-arg]
            probabilities, probability_targets
        )
        value_loss = self.value_criterion(values, value_targets)
        loss = policy_loss + value_loss
        accuracy_ = accuracy(probabilities, probability_targets)
        assert loss.shape == ()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return policy_loss, value_loss, accuracy_

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    @abstractmethod
    def states_to_tensor(self, states: Sequence[_S]) -> torch.Tensor:
        ...

    @abstractmethod
    def distributions_to_tensor(
        self, states: Sequence[_S], distributions: Sequence[Sequence[float]]
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def copy(self: _T) -> _T:
        ...

    def to(self: _T, device: torch.device) -> _T:
        self.device = device
        self.value_criterion.to(device)
        self.net.to(device)
        self.optimizer = self.optimizer_cls(  # type: ignore[call-arg]
            self.net.parameters(), *self.optimizer_args, **self.optimizer_kwargs
        )
        return self

    @classmethod
    def from_path(cls: Type[_T], path: str, *args: Any, **kwargs: Any) -> _T:
        anet = cls(*args, **kwargs)
        anet.load(path)
        return anet

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.net.load_state_dict(state_dict)


class GameNetAgent(Agent[_S]):
    net: GameNet[_S]

    def __init__(self, net: GameNet[_S]):
        self.net = net

    def reset(self) -> None:
        pass


class GreedyGameNetAgent(GameNetAgent[_S]):
    epsilon: float

    def __init__(self, net: GameNet[_S], epsilon: float = 0):
        super().__init__(net)
        self.epsilon = epsilon

    def play(self, state: _S) -> Action:
        return self.net.greedy_policy(state, self.epsilon)


class SamplingGameNetAgent(GameNetAgent[_S]):
    def play(self, state: _S) -> Action:
        return self.net.sampling_policy(state)


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


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        (
            torch.argmax(predictions.reshape((predictions.shape[0], -1)), dim=1)
            == torch.argmax(targets.reshape((targets.shape[0], -1)), dim=1)
        ).float()
    )
