from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    TensorModule = nn.Module[torch.Tensor]
    TensorPairModule = nn.Module[Tuple[torch.Tensor, torch.Tensor]]
else:
    TensorModule = nn.Module
    TensorPairModule = nn.Module


class ConvolutionalNet(TensorPairModule):
    def __init__(
        self,
        policy_features: int,
        policy_shape: Tuple[int, ...],
        grid_size: int,
        in_channels: int,
        num_residual: int,
        channels: int,
        value_head_hidden_units: int,
    ) -> None:
        super().__init__()
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels, out_channels=channels, kernel_size=3, padding=1
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
        self.policy_head = PolicyHead(
            grid_size, channels, policy_features, policy_shape
        )
        self.value_head = ValueHead(grid_size, channels, value_head_hidden_units)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = x
        # assert len(input.shape) == 4 and input.shape[1] == 3
        x = self.conv1(x)
        for (
            residual_block
        ) in (
            self.residual_blocks  # type: ignore[attr-defined] # https://github.com/pytorch/pytorch/pull/27445
        ):
            x = residual_block(x)
        value, probabilities = self.value_head(x), self.policy_head(x)
        probabilities = probabilities.squeeze(1)
        assert value.shape == (input.shape[0], 1)
        return value, probabilities


class ConvolutionalBlock(TensorModule):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class ResidualBlock(TensorModule):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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


class PolicyHead(TensorModule):
    def __init__(
        self,
        grid_size: int,
        in_channels: int,
        out_features: int,
        out_shape: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.out_shape = out_shape
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(
            in_features=in_channels * grid_size ** 2, out_features=out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.fc1(x.reshape((x.shape[0], -1))).reshape((-1, 1, *self.out_shape))
        return x


class ValueHead(TensorModule):
    def __init__(self, grid_size: int, in_channels: int, hidden_units: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalBlock(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )
        self.fc1 = nn.Linear(in_features=grid_size ** 2, out_features=hidden_units)
        self.fc2 = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv1(x)
        x = self.fc1(x.reshape((x.shape[0], -1)))
        x = F.relu(x)
        x = self.fc2(x)
        return x
