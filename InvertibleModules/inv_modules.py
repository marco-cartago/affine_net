import torch
import numpy as np
from typing import Tuple

from InvertibleModules.utils import *


class LUBlock(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        scale: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float
    ) -> None:

        super().__init__()

        # Init L and U matrices
        self.L = torch.nn.Parameter(torch.empty(dim, dim, dtype=dtype))
        self.U = torch.nn.Parameter(torch.empty(dim, dim, dtype=dtype))
        L_mask, _ = triangular_xavier_norm_(self.L, upper=False)
        U_mask, _ = triangular_xavier_norm_(self.U, upper=True)

        # L, U, L_mask, U_mask = triang_QR_gen_((dim, dim), dtype=dtype)
        # self.L = torch.nn.Parameter(L)
        # self.U = torch.nn.Parameter(U)

        self.L_hook = freeze_weights(self.L, L_mask)
        self.W_hook = freeze_weights(self.U, U_mask)

        # Init bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(dim, dtype=dtype))
            torch.nn.init.normal_(self.bias)

        # Scale parameters
        self.scale = None
        if scale:
            self.scale = torch.nn.Parameter(torch.empty(dim, dtype=dtype))
            torch.nn.init.normal_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.scale is not None:
            scale = self.scale
            x = torch.mul(x, scale)

        # (self.L @ self.U @ x.unsqueeze(dim=-1)).squeeze(-1)
        xp = torch.matmul(
            self.L,
            torch.matmul(self.U, x.unsqueeze(dim=-1))
        )

        xp = xp.squeeze(-1)
        if self.bias is not None:
            xp += self.bias

        return xp

    def inverse(self, x: torch.Tensor) -> torch.Tensor:

        if self.bias is not None:
            x = x - self.bias

        x = x.unsqueeze(-1)
        x = torch.linalg.solve_triangular(
            self.L, x, upper=False, unitriangular=True)
        x = torch.linalg.solve_triangular(
            self.U, x, upper=True,  unitriangular=True)
        x = x.squeeze(-1)

        if self.scale is not None:
            scale = self.scale
            x = x / scale

        return x


class ExtendDim(torch.nn.Module):

    def __init__(self, pad_dims: int, dtype: torch.dtype = torch.float) -> None:
        super().__init__()
        self.pad_dim = pad_dims
        self.dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n_obs, _ = input.shape
        zrs = torch.zeros(n_obs, self.pad_dim,
                          device=input.device, dtype=self.  dtype)
        out = torch.cat((input, zrs), dim=-1)
        return out

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        return input[..., :-self.pad_dim]


class I_LeakyReLU(torch.nn.Module):

    def __init__(self, negative_slope: float = 1e-2,
                 inplace: bool = True) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

        if np.isclose(self.negative_slope, 0):
            raise ValueError(
                "Negative slope is close to 0: {negative_slope}\n" +
                "Inverse might be unstable.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            input,
            self.negative_slope,
            self.inplace
        )

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        clone_input = input
        if not self.inplace:
            clone_input = input.clone()
        mask = clone_input < 0
        clone_input[mask] = clone_input[mask] / self.negative_slope
        return clone_input


class I_Cubic(torch.nn.Module):

    def __init__(self, slope=1e-1) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.slope * input, 3.0)

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        return torch.pow(input / self.slope, 1/3.0)


class I_SoftPlus(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(input))

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(input) - 1)
