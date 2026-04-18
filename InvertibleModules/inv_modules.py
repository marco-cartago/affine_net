import torch
import torch.nn.functional as F
import torch.linalg as L
import numpy as np

from typing import Tuple, Optional

from InvertibleModules.utils import *


class LUBlock(torch.nn.Module):
    """
    Linear layer parametrized as L·U with optional bias and per-feature scaling.
    L and U are forced to stay triangular via frozen-weight hooks.
    """

    def __init__(
        self,
        dim: int,
        scale: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()

        # L & U
        self.l = torch.nn.Parameter(torch.empty(dim, dim, dtype=dtype))
        self.u = torch.nn.Parameter(torch.empty(dim, dim, dtype=dtype))

        # Initialise as triangular matrices (Xavier‑scaled)
        L_mask, _ = triangular_xavier_norm_(self.l, upper=False)
        U_mask, _ = triangular_xavier_norm_(self.u, upper=True)

        # Freeze non‑triangular entries
        self.l_hook = freeze_weights(self.l, L_mask)
        self.u_hook = freeze_weights(self.u, U_mask)

        # bias 
        self.bias: Optional[torch.nn.Parameter] = None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(dim, dtype=dtype))
            torch.nn.init.normal_(self.bias)

        # scale 
        self.scale: Optional[torch.nn.Parameter] = None
        if scale:
            self.scale = torch.nn.Parameter(torch.empty(dim, dtype=dtype))
            torch.nn.init.normal_(self.scale)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Apply (scale)·L·U·x + bias."""
        
        if self.scale is not None:
            x = x * self.scale

        # L @ (U @ x)
        xp = torch.matmul(self.l,
                          torch.matmul(self.u, x.unsqueeze(-1))).squeeze(-1)

        if self.bias is not None:
            xp = xp + self.bias
        return xp

    def inverse(self, y: torch.Tensor) -> torch.Tensor:

        """Solve L·U·x = (y-bias)/scale for x."""
        
        if self.bias is not None:
            y = y - self.bias

        # solve U·x = z, then L·z = y
        z = L.solve_triangular(
            self.u, y.unsqueeze(-1), upper=True, unitriangular=True)
        x = L.solve_triangular(
            self.l, z, upper=False, unitriangular=True).squeeze(-1)

        if self.scale is not None:
            x = x / self.scale
            
        return x


class TwoChan(torch.nn.Module):
    """
    Simple reversible two-channel block.

    * `F` and `G` are any callable `torch.nn.Module`s.
    * inp/output are 2-tuples of tensors with identical shapes.
    """

    def __init__(self, F: torch.nn.Module, G: torch.nn.Module):
        super().__init__()               # initialise nn.Module
        self.F = F
        self.G = G

    def forward(self, inps: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        i1, i2 = inps
        if i1.shape != i2.shape:
            raise ValueError("i1 and i2 must have the same shape")

        o2 = i2 + self.F(i1)             
        o1 = i1 + self.G(o2)             
        return o1, o2

    def inverse(self, outputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        o1, o2 = outputs
        if o1.shape != o2.shape:
            raise ValueError("i1 and i2 must have the same shape")

        i1 = o1 - self.G(o2)            
        i2 = o2 - self.F(i1)

        return i1, i2

class ExtendDim(torch.nn.Module):
    """
    Appends `pad_dim` zero columns to the last dimension.
    Inverse removes the appended columns.
    """

    def __init__(self, pad_dim: int, dtype: torch.dtype = torch.float) -> None:
        super().__init__()
        self.pad_dim = pad_dim
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        zeros = torch.zeros(x.size(0), self.pad_dim,
                            device=x.device, dtype=self.dtype)
        return torch.cat((x, zeros), dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y[..., :-self.pad_dim]


class I_LeakyReLU(torch.nn.Module):

    """
    LeakyReLU with a deterministic inverse (when slope ≠ 0).
    """

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = True) -> None:
        super().__init__()
        if np.isclose(negative_slope, 0.0):
            raise ValueError(
                f"Negative slope is too close to 0 ({negative_slope}); inverse would be unstable."
            )
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # If not in‑place, work on a copy to avoid side‑effects
        out = y if self.inplace else y.clone()
        mask = out < 0
        out[mask] = out[mask] / self.negative_slope
        return out

class I_Cubic(torch.nn.Module):

    def __init__(self, slope=1e-1) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.slope * inp, 3.0)

    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.pow(inp / self.slope, 1/3.0)


class I_SoftPlus(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(inp))

    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(inp) - 1)
